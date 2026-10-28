#!/usr/bin/env python3
"""Extract compact replay facts from a line-oriented DLPTI native export."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


KIND_NAMES = {
    1: "DRIVER_API", 2: "RUNTIME_API", 20: "CU_KERNEL", 21: "TU_KERNEL",
    32: "GRAPH", 33: "GRAPH_INSTANCE", 49: "COMMAND", 50: "MEMCPY",
    56: "CU", 57: "FLUSH", 58: "TU",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    kind_counts: Counter[int] = Counter()
    kind_elapsed_ns: defaultdict[int, float] = defaultdict(float)
    graph_exec_counts: Counter[int] = Counter()
    graph_elapsed_ns: defaultdict[int, float] = defaultdict(float)
    first_graph_submit = None
    last_graph_submit = None

    inside = False
    with args.input.open() as stream:
        for line in stream:
            if not inside:
                if '"activities"' in line and "[" in line:
                    inside = True
                continue
            text = line.strip()
            if text.startswith("]"):
                break
            if text.endswith(","):
                text = text[:-1]
            if not text:
                continue
            obj = json.loads(text)
            kind = int(obj.get("kind", -1))
            kind_counts[kind] += 1
            if obj.get("elapsed") is not None:
                kind_elapsed_ns[kind] += float(obj["elapsed"])
            if kind == 32:
                exec_id = obj.get("graph_exec_id")
                if exec_id is not None:
                    key = int(exec_id)
                    graph_exec_counts[key] += 1
                    graph_elapsed_ns[key] += float(obj.get("elapsed", 0))
                    submit = obj.get("submit")
                    if submit is not None:
                        first_graph_submit = submit if first_graph_submit is None else min(first_graph_submit, submit)
                        last_graph_submit = submit if last_graph_submit is None else max(last_graph_submit, submit)

    repeats = Counter(graph_exec_counts.values())
    facts = {
        "input": str(args.input),
        "kind_counts": {KIND_NAMES.get(k, f"KIND_{k}"): v for k, v in sorted(kind_counts.items())},
        "kind_sum_ms": {KIND_NAMES.get(k, f"KIND_{k}"): v / 1e6 for k, v in sorted(kind_elapsed_ns.items())},
        "graph": {
            "launch_records": sum(graph_exec_counts.values()),
            "unique_graph_exec_ids": len(graph_exec_counts),
            "launches_per_exec_histogram": {str(k): v for k, v in sorted(repeats.items())},
            "most_replayed_exec_ids": [
                {"graph_exec_id": k, "launches": v, "sum_ms": graph_elapsed_ns[k] / 1e6}
                for k, v in graph_exec_counts.most_common(20)
            ],
            "first_submit_ns": first_graph_submit,
            "last_submit_ns": last_graph_submit,
        },
    }
    args.output.write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts["graph"], indent=2))


if __name__ == "__main__":
    main()
