#!/usr/bin/env python3
"""
Analyze benchmark results and generate comparison plots.
Reads JSON result files and produces p50/p95/p99 latency comparisons.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def compute_stats(results: list[dict]) -> dict:
    latencies = [r["latency_ms"] for r in results if r["status"] == 200]
    if not latencies:
        return {"count": 0}

    arr = np.array(latencies)
    errors = [r for r in results if r["status"] != 200]
    total_time = max(r["end_time"] for r in results) - min(r["start_time"] for r in results)

    return {
        "count": len(latencies),
        "errors": len(errors),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "throughput": len(latencies) / total_time if total_time > 0 else 0,
    }


def plot_latency_comparison(all_stats: dict[str, dict], output: Path):
    strategies = list(all_stats.keys())
    percentiles = ["p50", "p95", "p99"]
    x = np.arange(len(strategies))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, pctl in enumerate(percentiles):
        values = [all_stats[s].get(pctl, 0) for s in strategies]
        bars = ax.bar(x + i * width, values, width, label=pctl)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Routing Strategy")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Router Latency by Strategy (p50 / p95 / p99)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output / "latency_comparison.png", dpi=150)
    print(f"Saved latency comparison to {output / 'latency_comparison.png'}")


def plot_throughput_comparison(all_stats: dict[str, dict], output: Path):
    strategies = list(all_stats.keys())
    throughputs = [all_stats[s].get("throughput", 0) for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(strategies, throughputs, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom")

    ax.set_xlabel("Routing Strategy")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Router Throughput by Strategy")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output / "throughput_comparison.png", dpi=150)
    print(f"Saved throughput comparison to {output / 'throughput_comparison.png'}")


def plot_latency_distribution(all_results: dict[str, list[dict]], output: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (strategy, results) in enumerate(all_results.items()):
        if idx >= 4:
            break
        latencies = [r["latency_ms"] for r in results if r["status"] == 200]
        if not latencies:
            continue
        ax = axes[idx]
        ax.hist(latencies, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(np.percentile(latencies, 99), color="red", linestyle="--", label=f"p99={np.percentile(latencies, 99):.1f}ms")
        ax.axvline(np.percentile(latencies, 50), color="green", linestyle="--", label=f"p50={np.percentile(latencies, 50):.1f}ms")
        ax.set_title(strategy)
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    for idx in range(len(all_results), 4):
        axes[idx].set_visible(False)

    fig.suptitle("Latency Distribution by Strategy")
    fig.tight_layout()
    fig.savefig(output / "latency_distribution.png", dpi=150)
    print(f"Saved latency distribution to {output / 'latency_distribution.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("files", nargs="+", type=Path, help="Result JSON files (named strategy_results.json)")
    parser.add_argument("--output", type=Path, default=Path("plots"), help="Output directory for plots")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_stats = {}

    for f in args.files:
        strategy = f.stem.replace("_results", "").replace("results_", "")
        results = load_results(f)
        if results:
            strategy = results[0].get("strategy", strategy)
        all_results[strategy] = results
        all_stats[strategy] = compute_stats(results)

    print("\nStrategy Comparison:")
    print(f"{'Strategy':<20} {'Count':>8} {'Errors':>8} {'p50':>10} {'p95':>10} {'p99':>10} {'Throughput':>12}")
    print("-" * 80)
    for name, stats in all_stats.items():
        if stats["count"] == 0:
            print(f"{name:<20} {'no data':>8}")
            continue
        print(f"{name:<20} {stats['count']:>8} {stats['errors']:>8} {stats['p50']:>10.2f} {stats['p95']:>10.2f} {stats['p99']:>10.2f} {stats['throughput']:>10.1f}/s")

    if len(all_stats) >= 1:
        plot_latency_comparison(all_stats, args.output)
        plot_throughput_comparison(all_stats, args.output)
        plot_latency_distribution(all_results, args.output)


if __name__ == "__main__":
    main()
