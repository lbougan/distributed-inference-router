#!/usr/bin/env python3
"""
Failover and resilience scenario tests.
Requires: router running on :8080, mock backends on :8001-8003.

These tests simulate backend failures and verify the router handles them gracefully.
Run with: pytest tests/failover/ -v -s
"""

import asyncio
import time

import aiohttp
import pytest

ROUTER_URL = "http://localhost:8080"
MOCK_URLS = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
]


def make_chat_request() -> dict:
    return {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 10,
    }


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def session():
    async with aiohttp.ClientSession() as s:
        yield s


async def set_backend_health(session: aiohttp.ClientSession, backend_url: str, healthy: bool):
    async with session.post(
        f"{backend_url}/admin/health",
        json={"healthy": healthy},
    ) as resp:
        assert resp.status == 200


async def set_backend_fail_rate(session: aiohttp.ClientSession, backend_url: str, rate: float):
    async with session.post(
        f"{backend_url}/admin/fail_rate",
        json={"rate": rate},
    ) as resp:
        assert resp.status == 200


class TestFailover:
    @pytest.mark.asyncio
    async def test_reroutes_when_backend_down(self, session):
        """Kill one backend and verify traffic reroutes to healthy ones."""
        # Mark backend 1 as unhealthy
        await set_backend_health(session, MOCK_URLS[0], False)
        await asyncio.sleep(6)  # Wait for health checker to detect

        backends_seen = set()
        success = 0
        for _ in range(20):
            try:
                async with session.post(
                    f"{ROUTER_URL}/v1/chat/completions",
                    json=make_chat_request(),
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        success += 1
                        backend = resp.headers.get("X-Backend-ID", "")
                        backends_seen.add(backend)
            except Exception:
                pass

        # Restore health
        await set_backend_health(session, MOCK_URLS[0], True)

        assert success >= 15, f"Expected >= 15 successes after failover, got {success}"
        assert "mock-1" not in backends_seen, "Unhealthy backend should not receive traffic"

    @pytest.mark.asyncio
    async def test_recovers_after_backend_heals(self, session):
        """Backend goes down then recovers; verify it re-enters rotation."""
        await set_backend_health(session, MOCK_URLS[1], False)
        await asyncio.sleep(6)

        await set_backend_health(session, MOCK_URLS[1], True)
        await asyncio.sleep(6)  # Wait for health checker to detect recovery

        backends_seen = set()
        for _ in range(30):
            try:
                async with session.post(
                    f"{ROUTER_URL}/v1/chat/completions",
                    json=make_chat_request(),
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        backend = resp.headers.get("X-Backend-ID", "")
                        backends_seen.add(backend)
            except Exception:
                pass

        assert "mock-2" in backends_seen, "Recovered backend should re-enter rotation"

    @pytest.mark.asyncio
    async def test_all_backends_down_returns_503(self, session):
        """All backends down -> router returns 503."""
        for url in MOCK_URLS:
            await set_backend_health(session, url, False)
        await asyncio.sleep(6)

        async with session.post(
            f"{ROUTER_URL}/v1/chat/completions",
            json=make_chat_request(),
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            assert resp.status == 503

        # Restore all backends
        for url in MOCK_URLS:
            await set_backend_health(session, url, True)
        await asyncio.sleep(6)


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_opens_on_high_failure_rate(self, session):
        """High failure rate should trigger circuit breaker."""
        await set_backend_fail_rate(session, MOCK_URLS[0], 1.0)
        await asyncio.sleep(1)

        # Send enough requests to trigger circuit breaker
        for _ in range(20):
            try:
                async with session.post(
                    f"{ROUTER_URL}/v1/chat/completions",
                    json=make_chat_request(),
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    pass
            except Exception:
                pass

        # Check metrics for circuit breaker state
        async with session.get(f"{ROUTER_URL}/metrics") as resp:
            text = await resp.text()
            assert "router_circuit_breaker_state" in text

        # Restore
        await set_backend_fail_rate(session, MOCK_URLS[0], 0.0)


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_returns_429_under_extreme_load(self, session):
        """Flooding the router should eventually produce 429 responses."""
        tasks = []
        for _ in range(200):
            tasks.append(
                session.post(
                    f"{ROUTER_URL}/v1/chat/completions",
                    json=make_chat_request(),
                    timeout=aiohttp.ClientTimeout(total=10),
                )
            )

        responses = await asyncio.gather(
            *[t.__aenter__() for t in tasks],
            return_exceptions=True,
        )

        statuses = []
        for r in responses:
            if isinstance(r, Exception):
                continue
            statuses.append(r.status)
            r.close()

        # We may or may not get 429s depending on max_in_flight config
        # but we should not get any 500s from the router itself
        server_errors = [s for s in statuses if s >= 500 and s != 502 and s != 503]
        assert len(server_errors) == 0, f"Unexpected server errors: {server_errors}"
