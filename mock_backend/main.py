"""Mock vLLM server that mimics the OpenAI-compatible API surface, for development and
benchmarking without GPUs.

v2 behaves enough like a real replica for the routing strategies to be measurable:
- bounded concurrency with a real queue (``vllm:num_requests_waiting`` > 0 under load)
- KV cache usage that decays over time instead of only growing
- a prefix cache keyed on the prompt prefix, so hit rate depends on routing locality
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
import uuid
from collections import OrderedDict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

INSTANCE_ID = os.environ.get("INSTANCE_ID", "mock-0")
BASE_LATENCY_MS = float(os.environ.get("BASE_LATENCY_MS", "50"))
LATENCY_STDDEV_MS = float(os.environ.get("LATENCY_STDDEV_MS", "15"))
TOKENS_PER_REQUEST = int(os.environ.get("TOKENS_PER_REQUEST", "20"))
FAIL_RATE = float(os.environ.get("FAIL_RATE", "0.0"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "8"))
KV_DECAY_SECS = float(os.environ.get("KV_DECAY_SECS", "10"))
KV_PER_REQUEST = float(os.environ.get("KV_PER_REQUEST", "0.02"))
PREFIX_CACHE_SIZE = int(os.environ.get("PREFIX_CACHE_SIZE", "256"))
PREFIX_KEY_CHARS = 64


def _validate_settings() -> None:
    if MAX_CONCURRENT < 1:
        raise SystemExit(f"MAX_CONCURRENT must be >= 1, got {MAX_CONCURRENT}")
    if KV_DECAY_SECS <= 0:
        raise SystemExit(f"KV_DECAY_SECS must be > 0, got {KV_DECAY_SECS}")


_validate_settings()
PREFIX_HIT_SPEEDUP = 0.6  # a cache hit skips prefill: 40 % faster

SAMPLE_TOKENS = [
    "The",
    " answer",
    " to",
    " your",
    " question",
    " is",
    " that",
    " large",
    " language",
    " models",
    " work",
    " by",
    " predicting",
    " the",
    " next",
    " token",
    " in",
    " a",
    " sequence",
    ".",
]


class ChatMessage(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str = "mock-model"
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int = 100
    temperature: float = 0.7
    stream: bool = False
    n: int = 1


class PrefixCache:
    """LRU of prompt prefixes. ``lookup`` returns whether the prefix was already cached."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._entries: OrderedDict[str, None] = OrderedDict()
        self.hits = 0
        self.queries = 0

    def lookup(self, key: str) -> bool:
        self.queries += 1
        if key in self._entries:
            self._entries.move_to_end(key)
            self.hits += 1
            return True
        self._entries[key] = None
        if len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return False


class KvUsage:
    """KV cache occupancy in [0, 1] with exponential decay (half-life = decay_secs)."""

    def __init__(self, decay_secs: float):
        self.decay_secs = decay_secs
        self._value = 0.0
        self._ts = 0.0

    def current(self, now: float) -> float:
        elapsed = max(0.0, now - self._ts)
        return self._value * math.pow(0.5, elapsed / self.decay_secs)

    def touch(self, now: float, add: float) -> None:
        self._value = min(1.0, self.current(now) + add)
        self._ts = now


class Gate:
    """Bounded concurrency that counts how many requests are waiting for a slot."""

    def __init__(self, max_concurrent: int):
        self._sem = asyncio.Semaphore(max_concurrent)
        self.waiting = 0
        self.running = 0

    async def acquire(self) -> None:
        self.waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self.waiting -= 1
        self.running += 1

    def release(self) -> None:
        self.running -= 1
        self._sem.release()


gate = Gate(MAX_CONCURRENT)
prefix_cache = PrefixCache(PREFIX_CACHE_SIZE)
kv = KvUsage(KV_DECAY_SECS)

state = {
    "healthy": True,
    "fail_rate": FAIL_RATE,
    "total_requests": 0,
    "total_prompt_tokens": 0,
    "total_generation_tokens": 0,
}

app = FastAPI(title=f"Mock vLLM ({INSTANCE_ID})")


def prefix_key(req: CompletionRequest) -> str:
    if req.messages:
        text = " ".join(m.content for m in req.messages[:2])
    else:
        text = req.prompt or ""
    return text[:PREFIX_KEY_CHARS]


def simulate_latency() -> float:
    return max(5.0, random.gauss(BASE_LATENCY_MS, LATENCY_STDDEV_MS)) / 1000.0


def make_completion_response(request_id: str, tokens: list[str], is_chat: bool) -> dict:
    text = "".join(tokens)
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": len(tokens),
        "total_tokens": 10 + len(tokens),
    }
    if is_chat:
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [choice],
            "usage": usage,
        }
    choice = {"index": 0, "text": text, "finish_reason": "stop"}
    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": "mock-model",
        "choices": [choice],
        "usage": usage,
    }


def make_stream_chunk(request_id: str, token: str, is_chat: bool, finish: bool = False) -> str:
    if is_chat:
        choice = {
            "index": 0,
            "delta": {} if finish else {"content": token},
            "finish_reason": "stop" if finish else None,
        }
        obj = "chat.completion.chunk"
    else:
        choice = {
            "index": 0,
            "text": "" if finish else token,
            "finish_reason": "stop" if finish else None,
        }
        obj = "text_completion"
    chunk = {
        "id": request_id,
        "object": obj,
        "created": int(time.time()),
        "model": "mock-model",
        "choices": [choice],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def generate_stream(request_id: str, is_chat: bool, tokens: list[str]):
    for token in tokens:
        await asyncio.sleep(random.uniform(0.01, 0.05))
        yield make_stream_chunk(request_id, token, is_chat)
    yield make_stream_chunk(request_id, "", is_chat, finish=True)
    yield "data: [DONE]\n\n"


class GatedStreamingResponse(StreamingResponse):
    """Releases the concurrency slot when the response is finished or cancelled."""

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            gate.release()


async def handle_completion(req: CompletionRequest, is_chat: bool):
    await gate.acquire()
    released = False
    try:
        if random.random() < state["fail_rate"]:
            return JSONResponse(status_code=500, content={"error": "simulated failure"})

        request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        state["total_requests"] += 1
        hit = prefix_cache.lookup(prefix_key(req))
        kv.touch(time.monotonic(), KV_PER_REQUEST)

        latency = simulate_latency() * (PREFIX_HIT_SPEEDUP if hit else 1.0)
        await asyncio.sleep(latency)

        tokens = random.choices(SAMPLE_TOKENS, k=TOKENS_PER_REQUEST)
        state["total_prompt_tokens"] += 10
        state["total_generation_tokens"] += len(tokens)
        headers = {"X-Backend-ID": INSTANCE_ID, "X-Prefix-Cache": "hit" if hit else "miss"}

        if req.stream:
            released = True  # GatedStreamingResponse releases the gate when the stream ends
            return GatedStreamingResponse(
                generate_stream(request_id, is_chat, tokens),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", **headers},
            )
        return JSONResponse(
            content=make_completion_response(request_id, tokens, is_chat), headers=headers
        )
    finally:
        if not released:
            gate.release()


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    return await handle_completion(req, is_chat=False)


@app.post("/v1/chat/completions")
async def chat_completions(req: CompletionRequest):
    return await handle_completion(req, is_chat=True)


@app.get("/health")
async def health():
    if state["healthy"]:
        return {"status": "ok"}
    return JSONResponse(status_code=503, content={"status": "unhealthy"})


@app.post("/admin/health")
async def set_health(request: Request):
    """POST {"healthy": false} to simulate an outage."""
    body = await request.json()
    state["healthy"] = bool(body.get("healthy", True))
    return {"status": "ok", "healthy": state["healthy"]}


@app.post("/admin/fail_rate")
async def set_fail_rate(request: Request):
    """POST {"rate": 0.5} for 50 % HTTP 500s."""
    body = await request.json()
    state["fail_rate"] = float(body.get("rate", 0.0))
    return {"status": "ok", "fail_rate": state["fail_rate"]}


@app.get("/metrics")
async def metrics():
    label = INSTANCE_ID.replace("\\", "\\\\").replace('"', '\\"')
    inst = f'{{instance="{label}"}}'
    lines = [
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{inst} {gate.running}",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting{inst} {gate.waiting}",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        f"vllm:kv_cache_usage_perc{inst} {kv.current(time.monotonic()):.4f}",
        "# TYPE vllm:prefix_cache_hits counter",
        f"vllm:prefix_cache_hits{inst} {prefix_cache.hits}",
        "# TYPE vllm:prefix_cache_queries counter",
        f"vllm:prefix_cache_queries{inst} {prefix_cache.queries}",
        "# TYPE vllm:prompt_tokens_total counter",
        f"vllm:prompt_tokens_total{inst} {state['total_prompt_tokens']}",
        "# TYPE vllm:generation_tokens_total counter",
        f"vllm:generation_tokens_total{inst} {state['total_generation_tokens']}",
        "",
    ]
    return PlainTextResponse("\n".join(lines), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
