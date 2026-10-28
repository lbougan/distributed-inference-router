import asyncio

from benchmarks.load_test import _top_up, rate_limit_refiller


def test_refiller_releases_roughly_rps_times_elapsed():
    async def scenario() -> int:
        sem = asyncio.Semaphore(0)
        task = asyncio.create_task(rate_limit_refiller(sem, 100))
        await asyncio.sleep(0.5)
        task.cancel()
        return sem._value

    assert 35 <= asyncio.run(scenario()) <= 60


def test_top_up_keeps_pool_at_target():
    async def scenario() -> tuple[int, int]:
        tasks: set = set()

        def spawn() -> asyncio.Task:
            return asyncio.create_task(asyncio.sleep(0))

        for _ in range(5):
            spawn_task = spawn()
            tasks.add(spawn_task)
            spawn_task.add_done_callback(tasks.discard)
        _top_up(tasks, 30, spawn)
        first = len(tasks)
        await asyncio.sleep(0.01)  # let some/all complete
        _top_up(tasks, 30, spawn)
        return first, len(tasks)

    assert asyncio.run(scenario()) == (30, 30)
