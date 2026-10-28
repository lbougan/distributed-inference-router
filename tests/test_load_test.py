import asyncio
import random

import pytest

from benchmarks import load_test
from benchmarks.load_test import BenchmarkConfig, RequestResult, _top_up, rate_limit_refiller


@pytest.mark.parametrize(("rps", "secs", "low", "high"), [(100, 0.5, 45, 55), (30, 1.0, 27, 33)])
def test_refiller_releases_roughly_rps_times_elapsed(rps, secs, low, high):
    async def scenario() -> int:
        sem = asyncio.Semaphore(0)
        task = asyncio.create_task(rate_limit_refiller(sem, rps))
        await asyncio.sleep(secs)
        task.cancel()
        return sem._value

    assert low <= asyncio.run(scenario()) <= high


def test_top_up_keeps_pool_at_target():
    async def scenario() -> tuple[int, int]:
        tasks: set = set()

        def spawn() -> asyncio.Task:
            return asyncio.create_task(asyncio.sleep(0))

        for _ in range(5):
            tasks.add(spawn())
        _top_up(tasks, 30, spawn)
        first = len(tasks)
        await asyncio.sleep(0.01)  # let them complete
        tasks -= {t for t in tasks if t.done()}
        _top_up(tasks, 30, spawn)
        return first, len(tasks)

    assert asyncio.run(scenario()) == (30, 30)


def test_duration_loop_returns_each_result_exactly_once(monkeypatch):
    spawned: list[int] = []

    async def fake_send(session, config, semaphore, rate_limiter) -> RequestResult:
        await asyncio.sleep(random.uniform(0.001, 0.005))
        n = len(spawned)
        spawned.append(n)
        return RequestResult(
            start_time=n, end_time=n, latency_ms=1.0, status=200, backend="b", strategy="s"
        )

    monkeypatch.setattr(load_test, "send_request", fake_send)
    config = BenchmarkConfig(concurrency=20, duration_secs=0.3)
    results = asyncio.run(load_test.run_benchmark(config))
    seqs = [r.start_time for r in results]
    assert len(results) == len(spawned)
    assert len(set(seqs)) == len(seqs)
