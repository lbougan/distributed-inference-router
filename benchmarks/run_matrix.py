#!/usr/bin/env python3
"""Run the strategy x scenario benchmark matrix against the compose mock stack and write
benchmarks/results/<scenario>_<strategy>_r<repeat>.json. Then: python -m benchmarks.analyze
benchmarks/results/*_r0.json --markdown --update-readme README.md
The suggested analyze command uses repeat 0 only: run with --repeats 1 for a README table, or
pick the median repeat by hand.

Each (scenario, strategy) pair restarts the router container with a generated config so the
strategy and backpressure limits are exactly what the row claims.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import yaml

from benchmarks.load_test import BenchmarkConfig, run_benchmark, save_results

REPO = Path(__file__).resolve().parent.parent
COMPOSE = ["docker", "compose", "-f", str(REPO / "deploy" / "docker-compose.mock.yml")]
ROUTER_URL = "http://localhost:8080"
MOCK_URLS = ["http://localhost:8001", "http://localhost:8002", "http://localhost:8003"]
BACKENDS = ["http://mock-1:8000", "http://mock-2:8000", "http://mock-3:8000"]
RESULTS_DIR = REPO / "benchmarks" / "results"

log = logging.getLogger("run_matrix")


@dataclass(frozen=True)
class Scenario:
    name: str
    strategies: tuple[str, ...]
    load: dict = field(default_factory=dict)
    router: dict = field(default_factory=dict)
    kill_backend_at: float | None = None  # seconds into a duration run
    restore_backend_at: float | None = None
    repeats: int = 3


SCENARIOS = [
    Scenario(
        name="baseline",
        strategies=(
            "round_robin",
            "least_connections",
            "least_load",
            "latency_aware",
            "prefix_cache",
            # Router overhead = router_single_mock-2 minus direct_mock-2 at equal concurrency (16).
            # The multi-backend rows spread the load, so they are not comparable to `direct`.
            "direct",  # straight at mock-2, no router
            "router_single",  # router with only mock-2 behind it, round_robin
        ),
        load={"concurrency": 50, "total_requests": 2000},
    ),
    Scenario(
        name="prefix_locality",
        strategies=("round_robin", "prefix_cache"),
        load={"concurrency": 50, "total_requests": 2000, "prefix_pool": 10},
    ),
    Scenario(
        name="failover",
        strategies=("round_robin", "latency_aware"),
        load={"concurrency": 30, "duration_secs": 30, "rps_limit": 100},
        kill_backend_at=5.0,
        restore_backend_at=20.0,
        repeats=1,
    ),
    Scenario(
        name="saturation",
        strategies=("least_connections",),
        load={"concurrency": 500, "total_requests": 5000},
        router={"backpressure": {"max_in_flight": 100}},
        repeats=1,
    ),
]


def router_config(strategy: str, overrides: dict, backends: list[str] = BACKENDS) -> dict:
    cfg = {
        "listen": {"host": "0.0.0.0", "port": 8080},
        "routing": {"strategy": strategy, "prefix_cache": {"prefix_token_count": 256}},
        "backends": [{"url": u, "weight": 1} for u in backends],
        "health_check": {"interval_secs": 2, "unhealthy_threshold": 2, "healthy_threshold": 2},
        "circuit_breaker": {
            "failure_rate_threshold": 0.5,
            "min_requests": 5,
            "window_secs": 60,
            "open_duration_secs": 10,
        },
        "backpressure": {"max_in_flight": 1000, "max_backend_queue_depth": 50},
    }
    for section, values in overrides.items():
        cfg[section] = {**cfg.get(section, {}), **values}
    return cfg


def restart_router(config_path: Path) -> None:
    env = {**os.environ, "ROUTER_CONFIG": str(config_path)}
    subprocess.run(
        [*COMPOSE, "up", "-d", "--force-recreate", "--no-deps", "router"], env=env, check=True
    )


async def wait_healthy(session: aiohttp.ClientSession, url: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status == 200:
                    return
        except (aiohttp.ClientError, TimeoutError):
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError(f"{url} not healthy after {timeout_s}s")


async def set_mock_health(session: aiohttp.ClientSession, url: str, healthy: bool) -> None:
    async with session.post(f"{url}/admin/health", json={"healthy": healthy}) as resp:
        resp.raise_for_status()


async def mock_prefix_counters(session: aiohttp.ClientSession) -> dict[str, tuple[float, float]]:
    counters = {}
    for url in MOCK_URLS:
        async with session.get(f"{url}/metrics") as resp:
            text = await resp.text()
        hits = queries = 0.0
        for line in text.splitlines():
            if line.startswith("vllm:prefix_cache_hits{"):
                hits = float(line.split()[-1])
            elif line.startswith("vllm:prefix_cache_queries{"):
                queries = float(line.split()[-1])
        counters[url] = (hits, queries)
    return counters


SINGLE_BACKEND_CONCURRENCY = 16


async def run_one(scenario: Scenario, strategy: str, repeat: int) -> Path:
    direct = strategy == "direct"
    single = strategy == "router_single"
    label = {"direct": "direct_mock-2", "router_single": "router_single_mock-2"}.get(
        strategy, strategy
    )
    load = dict(scenario.load)
    if direct or single:
        load["concurrency"] = SINGLE_BACKEND_CONCURRENCY
    cfg = None
    if not direct:
        config_path = REPO / "benchmarks" / f".router-config.{scenario.name}.{strategy}.yaml"
        cfg = router_config(
            "round_robin" if single else strategy,
            scenario.router,
            [BACKENDS[1]] if single else BACKENDS,
        )
        config_path.write_text(yaml.safe_dump(cfg))
        restart_router(config_path)

    async with aiohttp.ClientSession() as session:
        if not direct:
            await wait_healthy(session, ROUTER_URL)
        for url in MOCK_URLS:
            await set_mock_health(session, url, True)
        if not direct:
            await asyncio.sleep(5)  # two health intervals so every backend is routable

        # Note: absolute p50s sit below the mocks' base latency because repeated prompts hit
        # the mock's prefix cache (~40% faster). That is also why baseline/round_robin p50 is
        # ~40 ms rather than 60 ms.
        bench = BenchmarkConfig(
            router_url=MOCK_URLS[1] if direct else ROUTER_URL,
            strategy=label,
            **load,
        )
        chaos = None
        if scenario.kill_backend_at is not None:

            async def chaos_task() -> None:
                await asyncio.sleep(scenario.kill_backend_at)
                log.info("killing %s", MOCK_URLS[0])
                await set_mock_health(session, MOCK_URLS[0], False)
                if scenario.restore_backend_at is not None:
                    await asyncio.sleep(scenario.restore_backend_at - scenario.kill_backend_at)
                    log.info("restoring %s", MOCK_URLS[0])
                    await set_mock_health(session, MOCK_URLS[0], True)

            chaos = asyncio.create_task(chaos_task())

        try:
            before = await mock_prefix_counters(session)
            results = await run_benchmark(bench)
            if chaos:
                await chaos
            after = await mock_prefix_counters(session)
        finally:
            if chaos:
                chaos.cancel()
                await asyncio.gather(chaos, return_exceptions=True)
            for url in MOCK_URLS:
                await set_mock_health(session, url, True)

    deltas = {u: (after[u][0] - before[u][0], after[u][1] - before[u][1]) for u in MOCK_URLS}
    total_q = sum(q for _, q in deltas.values())
    meta = {
        "scenario": scenario.name,
        "strategy": label,
        "repeat": repeat,
        "load": load,
        "router": scenario.router,
        "router_config": None if direct else cfg,
        "prefix_hit_rate": {u: (h / q if q else 0.0) for u, (h, q) in deltas.items()},
        "prefix_hit_rate_total": (sum(h for h, _ in deltas.values()) / total_q if total_q else 0.0),
        "router_commit": subprocess.run(
            ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=False
        ).stdout.strip(),
    }
    out = RESULTS_DIR / f"{scenario.name}_{strategy}_r{repeat}.json"
    save_results(results, out, meta)
    return out


async def main_async(only: set[str] | None, repeats_override: int | None) -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([*COMPOSE, "up", "-d", "--build"], check=True)
    outputs = []
    for scenario in SCENARIOS:
        if only and scenario.name not in only:
            continue
        repeats = repeats_override or scenario.repeats
        for strategy in scenario.strategies:
            for repeat in range(repeats):
                log.info("=== %s / %s (repeat %d/%d)", scenario.name, strategy, repeat + 1, repeats)
                outputs.append(await run_one(scenario, strategy, repeat))
    return outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Benchmark matrix runner")
    parser.add_argument("--only", nargs="*", help="Scenario names to run (default: all)")
    parser.add_argument("--repeats", type=int, help="Override repeats for every scenario")
    args = parser.parse_args()
    outputs = asyncio.run(main_async(set(args.only) if args.only else None, args.repeats))
    print("\nResults:")
    for p in outputs:
        print(f"  {p.relative_to(REPO)}")
    print(
        "\nNext: uv run python -m benchmarks.analyze benchmarks/results/*_r0.json --markdown --update-readme README.md"
    )


if __name__ == "__main__":
    main()
