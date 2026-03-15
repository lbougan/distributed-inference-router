"""
Mock vLLM server that mimics the OpenAI-compatible API surface.
Used for development and testing without requiring GPUs.
"""

import asyncio
import json
import os
import random
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

INSTANCE_ID = os.environ.get("INSTANCE_ID", "mock-0")
BASE_LATENCY_MS = float(os.environ.get("BASE_LATENCY_MS", "50"))
LATENCY_STDDEV_MS = float(os.environ.get("LATENCY_STDDEV_MS", "15"))
TOKENS_PER_REQUEST = int(os.environ.get("TOKENS_PER_REQUEST", "20"))
FAIL_RATE = float(os.environ.get("FAIL_RATE", "0.0"))

state = {
    "healthy": True,
    "requests_running": 0,
    "requests_waiting": 0,
    "total_requests": 0,
    "total_prompt_tokens": 0,
    "total_generation_tokens": 0,
    "kv_cache_usage": 0.0,
    "prefix_cache_hits": 0,
    "prefix_cache_queries": 0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title=f"Mock vLLM ({INSTANCE_ID})", lifespan=lifespan)


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


SAMPLE_TOKENS = [
    "The", " answer", " to", " your", " question", " is", " that",
    " large", " language", " models", " work", " by", " predicting",
    " the", " next", " token", " in", " a", " sequence", ".",
]


def simulate_latency() -> float:
    return max(5, random.gauss(BASE_LATENCY_MS, LATENCY_STDDEV_MS)) / 1000.0


def make_completion_response(request_id: str, tokens: list[str], is_chat: bool) -> dict:
    text = "".join(tokens)
    if is_chat:
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": len(tokens),
                "total_tokens": 10 + len(tokens),
            },
        }
    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": "mock-model",
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": len(tokens),
            "total_tokens": 10 + len(tokens),
        },
    }


def make_stream_chunk(request_id: str, token: str, is_chat: bool, finish: bool = False) -> str:
    if is_chat:
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "delta": {"content": token} if not finish else {},
                "finish_reason": "stop" if finish else None,
            }],
        }
    else:
        chunk = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "text": token if not finish else "",
                "finish_reason": "stop" if finish else None,
            }],
        }
    return f"data: {json.dumps(chunk)}\n\n"


async def generate_stream(request_id: str, is_chat: bool):
    tokens = random.choices(SAMPLE_TOKENS, k=TOKENS_PER_REQUEST)
    for token in tokens:
        inter_token_delay = random.uniform(0.01, 0.05)
        await asyncio.sleep(inter_token_delay)
        yield make_stream_chunk(request_id, token, is_chat)
    yield make_stream_chunk(request_id, "", is_chat, finish=True)
    yield "data: [DONE]\n\n"


async def handle_completion(req: CompletionRequest, is_chat: bool):
    if random.random() < FAIL_RATE:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=500, content={"error": "simulated failure"})

    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    state["requests_running"] += 1
    state["total_requests"] += 1
    state["prefix_cache_queries"] += 1
    if random.random() < 0.3:
        state["prefix_cache_hits"] += 1
    state["kv_cache_usage"] = min(1.0, state["kv_cache_usage"] + random.uniform(0.001, 0.01))

    try:
        latency = simulate_latency()
        await asyncio.sleep(latency)

        if req.stream:
            return StreamingResponse(
                generate_stream(request_id, is_chat),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Backend-ID": INSTANCE_ID,
                },
            )

        tokens = random.choices(SAMPLE_TOKENS, k=TOKENS_PER_REQUEST)
        state["total_prompt_tokens"] += 10
        state["total_generation_tokens"] += len(tokens)

        resp = make_completion_response(request_id, tokens, is_chat)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=resp,
            headers={"X-Backend-ID": INSTANCE_ID},
        )
    finally:
        state["requests_running"] = max(0, state["requests_running"] - 1)


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
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=503, content={"status": "unhealthy"})


@app.post("/admin/health")
async def set_health(request: Request):
    """Toggle health status. POST with {"healthy": false} to simulate failure."""
    body = await request.json()
    state["healthy"] = body.get("healthy", True)
    return {"status": "ok", "healthy": state["healthy"]}


@app.post("/admin/fail_rate")
async def set_fail_rate(request: Request):
    """Set failure rate. POST with {"rate": 0.5} for 50% failures."""
    global FAIL_RATE
    body = await request.json()
    FAIL_RATE = body.get("rate", 0.0)
    return {"status": "ok", "fail_rate": FAIL_RATE}


@app.get("/metrics")
async def metrics():
    lines = [
        "# HELP vllm:num_requests_running Number of requests currently running",
        "# TYPE vllm:num_requests_running gauge",
        f'vllm:num_requests_running{{instance="{INSTANCE_ID}"}} {state["requests_running"]}',
        "",
        "# HELP vllm:num_requests_waiting Number of requests waiting in queue",
        "# TYPE vllm:num_requests_waiting gauge",
        f'vllm:num_requests_waiting{{instance="{INSTANCE_ID}"}} {state["requests_waiting"]}',
        "",
        "# HELP vllm:kv_cache_usage_perc KV cache usage percentage",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        f'vllm:kv_cache_usage_perc{{instance="{INSTANCE_ID}"}} {state["kv_cache_usage"]:.4f}',
        "",
        "# HELP vllm:prefix_cache_hits Total prefix cache hits",
        "# TYPE vllm:prefix_cache_hits counter",
        f'vllm:prefix_cache_hits{{instance="{INSTANCE_ID}"}} {state["prefix_cache_hits"]}',
        "",
        "# HELP vllm:prefix_cache_queries Total prefix cache queries",
        "# TYPE vllm:prefix_cache_queries counter",
        f'vllm:prefix_cache_queries{{instance="{INSTANCE_ID}"}} {state["prefix_cache_queries"]}',
        "",
        "# HELP vllm:prompt_tokens_total Total prompt tokens processed",
        "# TYPE vllm:prompt_tokens_total counter",
        f'vllm:prompt_tokens_total{{instance="{INSTANCE_ID}"}} {state["total_prompt_tokens"]}',
        "",
        "# HELP vllm:generation_tokens_total Total generation tokens produced",
        "# TYPE vllm:generation_tokens_total counter",
        f'vllm:generation_tokens_total{{instance="{INSTANCE_ID}"}} {state["total_generation_tokens"]}',
        "",
    ]
    return PlainTextResponse("\n".join(lines), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
