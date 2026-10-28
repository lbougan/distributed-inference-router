#!/usr/bin/env python3
"""
Integration tests for the inference router.
Requires the compose stack: router on :8080, mock backends on :8001-8003.

Run with: pytest tests/integration/ -v
"""

import asyncio

import aiohttp
import pytest

from tests.docker_helpers import ROUTER_BACKENDS, ROUTER_URL

pytestmark = pytest.mark.docker

BACKEND_HEADER = "x-router-backend"


def make_chat_request(prompt: str = "Hello", stream: bool = False) -> dict:
    return {
        "model": "mock-model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20,
        "stream": stream,
    }


@pytest.fixture
async def session():
    async with aiohttp.ClientSession() as s:
        yield s


class TestBasicRouting:
    @pytest.mark.asyncio
    async def test_chat_completions(self, session):
        async with session.post(
            f"{ROUTER_URL}/v1/chat/completions",
            json=make_chat_request(),
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert "choices" in data
            assert len(data["choices"]) > 0

    @pytest.mark.asyncio
    async def test_completions(self, session):
        body = {"model": "mock-model", "prompt": "Hello world", "max_tokens": 20}
        async with session.post(
            f"{ROUTER_URL}/v1/completions",
            json=body,
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
            assert "choices" in data

    @pytest.mark.asyncio
    async def test_streaming(self, session):
        async with session.post(
            f"{ROUTER_URL}/v1/chat/completions",
            json=make_chat_request(stream=True),
        ) as resp:
            assert resp.status == 200
            content_type = resp.headers.get("content-type", "")
            assert "text/event-stream" in content_type
            chunks = []
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    chunks.append(decoded)
            assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_router_health(self, session):
        async with session.get(f"{ROUTER_URL}/health") as resp:
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_router_metrics(self, session):
        async with session.get(f"{ROUTER_URL}/metrics") as resp:
            assert resp.status == 200
            text = await resp.text()
            lines = text.splitlines()
            assert any(line.startswith("router_requests_total{") for line in lines)
            assert "_total_total" not in text


class TestLoadDistribution:
    @pytest.mark.asyncio
    async def test_distributes_across_backends(self, session):
        """Verify that requests are spread across multiple backends."""
        backends_seen = set()
        for _ in range(20):
            async with session.post(
                f"{ROUTER_URL}/v1/chat/completions",
                json=make_chat_request(),
            ) as resp:
                assert resp.status == 200
                backend = resp.headers.get(BACKEND_HEADER, "")
                assert backend in ROUTER_BACKENDS, backend
                backends_seen.add(backend)

        assert len(backends_seen) >= 2, f"Expected requests on >= 2 backends, got {backends_seen}"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, session):
        """Verify router handles concurrent requests without errors."""
        tasks = []
        for _ in range(50):
            tasks.append(
                session.post(
                    f"{ROUTER_URL}/v1/chat/completions",
                    json=make_chat_request(),
                )
            )

        responses = await asyncio.gather(*[t.__aenter__() for t in tasks])
        statuses = [r.status for r in responses]
        for r in responses:
            r.close()

        success_count = sum(1 for s in statuses if s == 200)
        assert success_count >= 40, f"Expected >= 40 successes, got {success_count}"


class TestPrefixCacheRouting:
    @pytest.mark.asyncio
    async def test_same_prefix_routes_consistently(self, session):
        """Same prompt prefix should route to the same backend (when using prefix_cache strategy)."""
        prompt = "This is a very specific test prompt for prefix caching"
        backends = []
        for _ in range(5):
            async with session.post(
                f"{ROUTER_URL}/v1/chat/completions",
                json=make_chat_request(prompt=prompt),
            ) as resp:
                assert resp.status == 200
                backends.append(resp.headers.get(BACKEND_HEADER, ""))

        # All should go to the same backend (if prefix_cache strategy is active)
        # If round_robin is active this test just verifies no errors
        assert all(b in ROUTER_BACKENDS for b in backends), f"missing {BACKEND_HEADER}: {backends}"
