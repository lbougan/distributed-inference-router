"""Helpers shared by the docker-only tests (router on :8080, mocks on :8001-8003)."""

import asyncio
import re
import time

import aiohttp

ROUTER_URL = "http://localhost:8080"
# Host-side admin URLs and the names the router knows the same backends by.
MOCK_ADMIN_URLS = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
]
ROUTER_BACKENDS = [
    "http://mock-1:8000",
    "http://mock-2:8000",
    "http://mock-3:8000",
]
# deploy/router-config.yaml: 5 s interval, thresholds 3/2, so detection takes ~15 s.
HEALTH_TIMEOUT_SECS = 30.0
POLL_INTERVAL_SECS = 0.5


def metric_sample(text: str, name: str, backend_url: str) -> float | None:
    """Value of `name{backend="<backend_url>"}` in Prometheus text, or None."""
    pattern = rf'^{re.escape(name)}\{{backend="{re.escape(backend_url)}"\}} (\S+)$'
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else None


async def router_metrics(session: aiohttp.ClientSession) -> str:
    async with session.get(f"{ROUTER_URL}/metrics") as resp:
        assert resp.status == 200
        return await resp.text()


async def wait_for_backend_health(
    session: aiohttp.ClientSession,
    backend_url: str,
    expected: int,
    timeout: float = HEALTH_TIMEOUT_SECS,
) -> None:
    """Poll the router's `router_backend_health` gauge until it equals `expected` (0 or 1)."""
    if expected not in (0, 1):
        raise ValueError(f"expected must be 0 or 1, got {expected}")
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = metric_sample(await router_metrics(session), "router_backend_health", backend_url)
        if last == expected:
            return
        await asyncio.sleep(POLL_INTERVAL_SECS)
    raise AssertionError(
        f"router_backend_health for {backend_url} still {last} after {timeout}s, wanted {expected}"
    )


async def set_backend_health(session: aiohttp.ClientSession, admin_url: str, healthy: bool):
    async with session.post(f"{admin_url}/admin/health", json={"healthy": healthy}) as resp:
        assert resp.status == 200


async def set_backend_fail_rate(session: aiohttp.ClientSession, admin_url: str, rate: float):
    async with session.post(f"{admin_url}/admin/fail_rate", json={"rate": rate}) as resp:
        assert resp.status == 200
