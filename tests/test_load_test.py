import asyncio

from benchmarks.load_test import rate_limit_refiller


def test_refiller_approximates_configured_rps():
    async def scenario() -> int:
        sem = asyncio.Semaphore(0)
        task = asyncio.create_task(rate_limit_refiller(sem, 100))
        await asyncio.sleep(0.5)
        task.cancel()
        return sem._value

    assert 35 <= asyncio.run(scenario()) <= 60
