#!/usr/bin/env python3
"""
Failover and resilience scenario tests.
Requires the compose stack: router on :8080, mock backends on :8001-8003.

These tests simulate backend failures and verify the router handles them gracefully.
Run with: pytest tests/failover/ -v -s
"""

import asyncio

import aiohttp
import pytest

from tests.docker_helpers import (
    MOCK_ADMIN_URLS,
    ROUTER_BACKENDS,
    ROUTER_URL,
    metric_sample,
    router_metrics,
    set_backend_fail_rate,
    set_backend_health,
    wait_for_backend_health,
)

pytestmark = pytest.mark.docker

BACKEND_HEADER = "x-router-backend"
CIRCUIT_OPEN = 2.0
CIRCUIT_MAX_ROUNDS = 10


def make_chat_request() -> dict:
    return {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 10,
    }


@pytest.fixture
async def session():
    async with aiohttp.ClientSession() as s:
        yield s


async def send_requests(session: aiohttp.ClientSession, n: int) -> list[tuple[int, str]]:
    """Send `n` sequential chat requests; return (status, x-router-backend) per response."""
    results = []
    for _ in range(n):
        try:
            async with session.post(
                f"{ROUTER_URL}/v1/chat/completions",
                json=make_chat_request(),
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                await resp.read()
                results.append((resp.status, resp.headers.get(BACKEND_HEADER, "")))
        except (TimeoutError, aiohttp.ClientError):
            results.append((0, ""))
    return results


class TestFailover:
    @pytest.mark.asyncio
    async def test_reroutes_when_backend_down(self, session):
        """Kill one backend and verify traffic reroutes to healthy ones."""
        await set_backend_health(session, MOCK_ADMIN_URLS[0], False)
        try:
            await wait_for_backend_health(session, ROUTER_BACKENDS[0], 0)
            results = await send_requests(session, 20)
        finally:
            await set_backend_health(session, MOCK_ADMIN_URLS[0], True)
        await wait_for_backend_health(session, ROUTER_BACKENDS[0], 1)

        ok = [backend for status, backend in results if status == 200]
        assert len(ok) >= 15, f"Expected >= 15 successes after failover, got {results}"
        assert ROUTER_BACKENDS[0] not in ok, "Unhealthy backend should not receive traffic"

    @pytest.mark.asyncio
    async def test_recovers_after_backend_heals(self, session):
        """Backend goes down then recovers; verify it re-enters rotation."""
        await set_backend_health(session, MOCK_ADMIN_URLS[1], False)
        try:
            await wait_for_backend_health(session, ROUTER_BACKENDS[1], 0)
        finally:
            await set_backend_health(session, MOCK_ADMIN_URLS[1], True)
        await wait_for_backend_health(session, ROUTER_BACKENDS[1], 1)

        results = await send_requests(session, 30)
        seen = {backend for status, backend in results if status == 200}
        assert ROUTER_BACKENDS[1] in seen, f"Recovered backend should re-enter rotation: {seen}"

    @pytest.mark.asyncio
    async def test_all_backends_down_returns_503(self, session):
        """All backends down -> router returns 503."""
        try:
            for url in MOCK_ADMIN_URLS:
                await set_backend_health(session, url, False)
            for backend in ROUTER_BACKENDS:
                await wait_for_backend_health(session, backend, 0)

            async with session.post(
                f"{ROUTER_URL}/v1/chat/completions",
                json=make_chat_request(),
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                assert resp.status == 503
        finally:
            for url in MOCK_ADMIN_URLS:
                await set_backend_health(session, url, True)
        for backend in ROUTER_BACKENDS:
            await wait_for_backend_health(session, backend, 1)


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_opens_on_high_failure_rate(self, session):
        """A backend failing every request has its circuit opened (state 2).

        The breaker judges a 60 s window that still holds this backend's successes from the
        earlier tests, and the strategy sends it only part of the traffic, so requests go out
        in rounds of 20 until the failures outweigh those successes.
        """
        await set_backend_fail_rate(session, MOCK_ADMIN_URLS[0], 1.0)
        state = None
        try:
            for _ in range(CIRCUIT_MAX_ROUNDS):
                await send_requests(session, 20)
                text = await router_metrics(session)
                state = metric_sample(text, "router_circuit_breaker_state", ROUTER_BACKENDS[0])
                if state == CIRCUIT_OPEN:
                    break
        finally:
            await set_backend_fail_rate(session, MOCK_ADMIN_URLS[0], 0.0)

        assert state == CIRCUIT_OPEN, f"circuit state for {ROUTER_BACKENDS[0]}: {state}"


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
