#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--label", default="vllm_graph")
    args = parser.parse_args()

    runs = []
    for path in sorted(args.results.glob("vllm-run*.json")):
        if path.stem.endswith("-warmup"):
            continue
        runs.append(json.loads(path.read_text())["measured"])
    if not runs:
        raise SystemExit("no measured vLLM runs found")

    facts = {
        "label": args.label,
        "runs": len(runs),
        "median_makespan_s": statistics.median(x["makespan_s"] for x in runs),
        "mean_makespan_s": statistics.fmean(x["makespan_s"] for x in runs),
        "median_output_tokens_per_s": statistics.median(x["output_tokens_per_s"] for x in runs),
        "median_p50_ttft_ms": statistics.median(x["ttft_ms"]["p50"] for x in runs),
        "median_request_latency_ms": statistics.median(x["request_latency_ms"]["p50"] for x in runs),
        "output_chunks_per_run": sorted({x["output_chunks"] for x in runs}),
        "per_run": runs,
    }
    (args.results / "target-facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps({k: v for k, v in facts.items() if k != "per_run"}, indent=2))


if __name__ == "__main__":
    main()
