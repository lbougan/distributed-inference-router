#!/usr/bin/env python3
"""
Async load generator for benchmarking the inference router.
Sends concurrent requests and records per-request latency, status, and backend.
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp
import yaml


@dataclass
class RequestResult:
    start_time: float
    end_time: float
    latency_ms: float
    status: int
    backend: str
    strategy: str
    error: str | None = None


@dataclass
class BenchmarkConfig:
    router_url: str = "http://localhost:8080"
    concurrency: int = 50
    total_requests: int = 1000
    rps_limit: float = 0.0  # 0 = unlimited
    duration_secs: float = 0.0  # 0 = use total_requests instead
    strategy: str = "round_robin"
    stream: bool = False
    prefix_pool: int = 0  # distinct system prompts; 0 = the single default
    prompt_pool: list[str] = field(
        default_factory=lambda: [
            "Explain quantum computing in simple terms.",
            "Write a Python function to sort a list.",
            "What is the capital of France?",
            "Describe the process of photosynthesis.",
            "How do neural networks learn?",
            "What are the benefits of microservices?",
            "Explain the CAP theorem.",
            "Write a haiku about programming.",
            "What is Rust's ownership model?",
            "Describe how HTTP/2 multiplexing works.",
        ]
    )


async def send_request(
    session: aiohttp.ClientSession,
    config: BenchmarkConfig,
    semaphore: asyncio.Semaphore,
    rate_limiter: asyncio.Semaphore | None,
) -> RequestResult:
    prompt = random.choice(config.prompt_pool)
    system = "You are a helpful assistant."
    if config.prefix_pool > 0:
        system = f"You are assistant variant {random.randrange(config.prefix_pool)}. Be concise."
    body = {
        "model": "mock-model",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "max_tokens": 50,
        "stream": config.stream,
    }

    async with semaphore:
        if rate_limiter:
            await rate_limiter.acquire()

        start = time.monotonic()
        try:
            async with session.post(
                f"{config.router_url}/v1/chat/completions",
                json=body,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if config.stream:
                    async for _ in resp.content:
                        pass
                else:
                    await resp.read()

                end = time.monotonic()
                backend = resp.headers.get("x-router-backend", "unknown")
                return RequestResult(
                    start_time=start,
                    end_time=end,
                    latency_ms=(end - start) * 1000,
                    status=resp.status,
                    backend=backend,
                    strategy=config.strategy,
                )
        except Exception as e:
            end = time.monotonic()
            return RequestResult(
                start_time=start,
                end_time=end,
                latency_ms=(end - start) * 1000,
                status=0,
                backend="error",
                strategy=config.strategy,
                error=str(e),
            )


async def rate_limit_refiller(rate_limiter: asyncio.Semaphore, rps: float):
    """Refills the rate limiter at the configured RPS."""
    interval = 1.0 / rps
    while True:
        await asyncio.sleep(interval)
        try:
            rate_limiter.release()
        except ValueError:
            pass


async def run_benchmark(config: BenchmarkConfig) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(config.concurrency)
    rate_limiter = None
    refiller_task = None

    if config.rps_limit > 0:
        rate_limiter = asyncio.Semaphore(0)
        refiller_task = asyncio.create_task(rate_limit_refiller(rate_limiter, config.rps_limit))

    connector = aiohttp.TCPConnector(limit=config.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        if config.duration_secs > 0:
            results = []
            deadline = time.monotonic() + config.duration_secs
            tasks = set()

            while time.monotonic() < deadline:
                if len(tasks) < config.concurrency * 2:
                    task = asyncio.create_task(
                        send_request(session, config, semaphore, rate_limiter)
                    )
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)

                done, _ = await asyncio.wait(
                    tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    results.append(t.result())

            if tasks:
                done, _ = await asyncio.wait(tasks, timeout=10)
                for t in done:
                    results.append(t.result())
        else:
            tasks = [
                send_request(session, config, semaphore, rate_limiter)
                for _ in range(config.total_requests)
            ]
            results = await asyncio.gather(*tasks)
            results = list(results)

    if refiller_task:
        refiller_task.cancel()

    return results


def results_to_dicts(results: list[RequestResult]) -> list[dict]:
    return [asdict(r) for r in results]


def save_results(results: list[RequestResult], output: Path, meta: dict | None = None) -> None:
    payload = {"meta": meta or {}, "results": results_to_dicts(results)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"Saved {len(results)} results to {output}")


def print_summary(results: list[RequestResult]):
    import numpy as np

    latencies = [r.latency_ms for r in results if r.status == 200]
    errors = [r for r in results if r.status != 200]

    if not latencies:
        print("No successful requests!")
        return

    arr = np.array(latencies)
    print(f"\n{'=' * 60}")
    print(f"  Benchmark Summary ({len(results)} total requests)")
    print(f"{'=' * 60}")
    print(f"  Successful: {len(latencies)}  Errors: {len(errors)}")
    print("  Latency (ms):")
    print(f"    p50:  {np.percentile(arr, 50):.2f}")
    print(f"    p90:  {np.percentile(arr, 90):.2f}")
    print(f"    p95:  {np.percentile(arr, 95):.2f}")
    print(f"    p99:  {np.percentile(arr, 99):.2f}")
    print(f"    mean: {arr.mean():.2f}")
    print(f"    min:  {arr.min():.2f}")
    print(f"    max:  {arr.max():.2f}")

    backends = {}
    for r in results:
        if r.status == 200:
            backends.setdefault(r.backend, []).append(r.latency_ms)

    if backends:
        print("\n  Per-backend breakdown:")
        for name, lats in sorted(backends.items()):
            ba = np.array(lats)
            print(
                f"    {name}: n={len(lats)} p50={np.percentile(ba, 50):.1f}ms p99={np.percentile(ba, 99):.1f}ms"
            )

    total_time = max(r.end_time for r in results) - min(r.start_time for r in results)
    print(f"\n  Throughput: {len(latencies) / total_time:.1f} req/s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"{'=' * 60}\n")


def load_scenario(path: Path) -> BenchmarkConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return BenchmarkConfig(
        **{k: v for k, v in data.items() if k in BenchmarkConfig.__dataclass_fields__}
    )


def main():
    parser = argparse.ArgumentParser(description="Inference Router Load Tester")
    parser.add_argument("--url", default="http://localhost:8080", help="Router URL")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument(
        "--duration", type=float, default=0, help="Duration in seconds (overrides --requests)"
    )
    parser.add_argument("--rps", type=float, default=0, help="Requests per second limit")
    parser.add_argument("--strategy", default="round_robin")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--prefix-pool", type=int, default=0, help="Distinct system prompts")
    parser.add_argument("--scenario", type=Path, help="YAML scenario file")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    if args.scenario:
        config = load_scenario(args.scenario)
    else:
        config = BenchmarkConfig(
            router_url=args.url,
            concurrency=args.concurrency,
            total_requests=args.requests,
            rps_limit=args.rps,
            duration_secs=args.duration,
            strategy=args.strategy,
            stream=args.stream,
            prefix_pool=args.prefix_pool,
        )

    print(f"Starting benchmark: {config.concurrency} concurrent, strategy={config.strategy}")
    results = asyncio.run(run_benchmark(config))
    save_results(
        results,
        args.output,
        {
            "strategy": config.strategy,
            "scenario": args.scenario.stem if args.scenario else "cli",
        },
    )
    print_summary(results)


if __name__ == "__main__":
    main()
