import json
from pathlib import Path

import pytest

from benchmarks import analyze


def _run(strategy: str, scenario: str, latencies: list[float], statuses: list[int]) -> dict:
    results = [
        {
            "start_time": i,
            "end_time": i + lat / 1000,
            "latency_ms": lat,
            "status": st,
            "backend": f"http://mock-{(i % 2) + 1}:8000",
            "strategy": strategy,
            "error": None,
        }
        for i, (lat, st) in enumerate(zip(latencies, statuses, strict=True))
    ]
    return {"meta": {"strategy": strategy, "scenario": scenario}, "results": results}


def test_compute_stats_counts_statuses_and_backend_shares():
    run = _run("rr", "baseline", [10, 20, 30, 40], [200, 200, 429, 502])
    stats = analyze.compute_stats(run["results"])
    assert stats["count"] == 2
    assert stats["rejected_429"] == 1
    assert stats["errors_5xx"] == 1
    assert stats["p50"] == 15.0
    assert stats["backend_share"] == {"http://mock-1:8000": 0.5, "http://mock-2:8000": 0.5}


def test_load_results_accepts_bare_list_and_envelope(tmp_path: Path):
    bare = tmp_path / "bare.json"
    bare.write_text(
        json.dumps(
            [{"latency_ms": 1, "status": 200, "start_time": 0, "end_time": 0.001, "backend": "b"}]
        )
    )
    env = tmp_path / "env.json"
    env.write_text(json.dumps(_run("rr", "baseline", [1.0], [200])))
    assert analyze.load_run(bare)["meta"] == {"strategy": "bare", "scenario": "unknown"}
    assert analyze.load_run(env)["meta"]["scenario"] == "baseline"


def test_markdown_table_has_one_row_per_run():
    runs = [
        _run("round_robin", "baseline", [10, 12], [200, 200]),
        _run("latency_aware", "baseline", [8, 9], [200, 200]),
    ]
    table = analyze.markdown_table(runs)
    lines = table.strip().splitlines()
    assert lines[0].startswith("| scenario | strategy |")
    assert len(lines) == 4  # header, separator, 2 rows
    assert lines[0].endswith("| prefix hit |")
    assert "| baseline | round_robin |" in lines[2]
    assert lines[2].endswith("| - |")


def test_update_readme_replaces_only_the_marked_block(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- BENCH:START -->\nold\n<!-- BENCH:END -->\noutro\n")
    analyze.update_readme(readme, "| a | b |\n", commit="abc1234")
    text = readme.read_text()
    assert text.startswith("intro\n<!-- BENCH:START -->\n")
    assert "old" not in text
    assert "abc1234" in text
    assert text.endswith("<!-- BENCH:END -->\noutro\n")


def test_update_readme_rejects_missing_markers(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n")
    with pytest.raises(ValueError, match="BENCH:START"):
        analyze.update_readme(readme, "| a |\n", commit="abc1234")
