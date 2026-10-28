#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


MODES = ("eager", "full_and_piecewise")


def load_runs(results: Path, mode: str) -> list[dict]:
    runs = []
    for path in sorted(results.glob(f"{mode}-run*.json")):
        if path.stem.endswith("-warmup"):
            continue
        obj = json.loads(path.read_text())
        runs.append(obj["measured"])
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    results = Path(args.results)
    aggregate = {}
    for mode in MODES:
        runs = load_runs(results, mode)
        if not runs:
            continue
        aggregate[mode] = {
            "runs": len(runs),
            "median_makespan_s": statistics.median(run["makespan_s"] for run in runs),
            "mean_makespan_s": statistics.fmean(run["makespan_s"] for run in runs),
            "median_output_tokens_per_s": statistics.median(run["output_tokens_per_s"] for run in runs),
            "median_p50_ttft_ms": statistics.median(run["ttft_ms"]["p50"] for run in runs),
            "constant_output_chunks": sorted({run["output_chunks"] for run in runs}),
        }

    comparison = {}
    if set(aggregate) == set(MODES):
        eager = aggregate["eager"]
        graph = aggregate["full_and_piecewise"]
        comparison = {
            "speedup_median_makespan": eager["median_makespan_s"] / graph["median_makespan_s"],
            "speedup_median_throughput": graph["median_output_tokens_per_s"] / eager["median_output_tokens_per_s"],
            "graph_ttft_ratio_at_p50": graph["median_p50_ttft_ms"] / eager["median_p50_ttft_ms"],
        }

    report = {"aggregate": aggregate, "comparison": comparison}
    (results / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
