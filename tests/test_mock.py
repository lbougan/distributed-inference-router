"""Unit and ASGI tests for the mock vLLM backend."""

import asyncio
import re

import httpx
import pytest

from mock_backend import main as mock


def test_prefix_cache_hits_only_on_repeated_prefix():
    cache = mock.PrefixCache(capacity=2)
    assert cache.lookup("system A | hello") is False
    assert cache.lookup("system A | hello") is True
    assert cache.lookup("system B | hello") is False
    assert cache.lookup("system C | hello") is False  # evicts A
    assert cache.lookup("system A | hello") is False
    assert cache.queries == 5
    assert cache.hits == 1


def test_kv_usage_decays_between_requests():
    kv = mock.KvUsage(decay_secs=10.0)
    kv.touch(now=0.0, add=0.5)
    assert kv.current(now=0.0) == pytest.approx(0.5)
    assert kv.current(now=10.0) == pytest.approx(0.5 * 0.5)  # one half-life
    kv.touch(now=10.0, add=0.9)
    assert kv.current(now=10.0) == 1.0  # clamped


def test_prefix_key_uses_system_and_first_user_message():
    req = mock.CompletionRequest(
        messages=[
            mock.ChatMessage(role="system", content="You are terse."),
            mock.ChatMessage(role="user", content="Explain CAP."),
        ]
    )
    assert mock.prefix_key(req) == "You are terse. Explain CAP."[: mock.PREFIX_KEY_CHARS]
    assert mock.prefix_key(mock.CompletionRequest(prompt="x" * 500)) == "x" * mock.PREFIX_KEY_CHARS


def _metric(text: str, name: str) -> float:
    match = re.search(rf"^{re.escape(name)}\{{[^}}]*\}} ([0-9.]+)$", text, re.M)
    assert match, f"{name} missing in:\n{text}"
    return float(match.group(1))


@pytest.mark.asyncio
async def test_queue_depth_is_visible_while_requests_wait(monkeypatch):
    monkeypatch.setattr(mock, "gate", mock.Gate(max_concurrent=1))
    monkeypatch.setattr(mock, "BASE_LATENCY_MS", 200.0)
    monkeypatch.setattr(mock, "LATENCY_STDDEV_MS", 0.0)
    transport = httpx.ASGITransport(app=mock.app)
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        posts = [
            asyncio.create_task(client.post("/v1/chat/completions", json=body)) for _ in range(3)
        ]
        await asyncio.sleep(0.05)
        metrics = (await client.get("/metrics")).text
        assert _metric(metrics, "vllm:num_requests_waiting") >= 1
        assert _metric(metrics, "vllm:num_requests_running") == 1
        responses = await asyncio.gather(*posts)
        assert all(r.status_code == 200 for r in responses)
        metrics = (await client.get("/metrics")).text
        assert _metric(metrics, "vllm:num_requests_waiting") == 0
        assert _metric(metrics, "vllm:prefix_cache_hits") == 2
        assert _metric(metrics, "vllm:prefix_cache_queries") == 3


@pytest.mark.asyncio
async def test_admin_endpoints_toggle_health_and_failures():
    transport = httpx.ASGITransport(app=mock.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        assert (await client.get("/health")).status_code == 200
        await client.post("/admin/health", json={"healthy": False})
        assert (await client.get("/health")).status_code == 503
        await client.post("/admin/health", json={"healthy": True})
        await client.post("/admin/fail_rate", json={"rate": 1.0})
        r = await client.post("/v1/completions", json={"model": "m", "prompt": "x"})
        assert r.status_code == 500
        await client.post("/admin/fail_rate", json={"rate": 0.0})
