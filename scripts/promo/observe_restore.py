#!/usr/bin/env python3
"""Run candidate-only P64 wall and refresh the frozen-baseline compare."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(os.environ.get("AUTOHELIX_ROOT", Path.cwd()))
SUITE = Path(
    os.environ.get(
        "SUITE_DIR",
        ROOT / "artifacts/promo/suite",
    )
)
BASELINE = SUITE / "baseline/vllm/summary.json"
URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL", "Qwen3-8B")
PARALLEL = int(os.environ.get("PARALLEL", "64"))
NUMBER = int(os.environ.get("NUMBER", "64"))


def percentile(path: Path, name: str, key: str) -> float:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return float(next(row[key] for row in rows if row["Percentiles"] == name))


def main() -> int:
    baseline_doc = json.loads(BASELINE.read_text(encoding="utf-8"))
    run_dir = SUITE / "candidate/infinilm/restore_live"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        subprocess.run(["rm", "-rf", str(run_dir)], check=True)
    run_dir.mkdir(parents=True)

    env = os.environ.copy()
    env.update(
        OPENAI_BASE_URL=URL,
        CLIENT_WINDOW_OUT=str(run_dir),
        MODEL=MODEL,
        PARALLEL=str(PARALLEL),
        NUMBER=str(NUMBER),
    )
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts/promo/client_window_evalscope_p64.sh")],
        env=env,
    )
    if proc.returncode != 0:
        print("[observe_restore] client wall failed", file=sys.stderr)
        return proc.returncode

    results = sorted(run_dir.rglob("benchmark_summary.json"))
    if not results:
        print("[observe_restore] benchmark summary missing", file=sys.stderr)
        return 2
    summary = json.loads(results[-1].read_text(encoding="utf-8"))
    percentile_path = results[-1].with_name("benchmark_percentile.json")
    tok_s = float(summary["Output Throughput (tok/s)"])
    success = float(summary["Success Requests"]) / float(summary["Total Requests"])
    ttft_p95 = percentile(percentile_path, "95%", "TTFT (ms)")
    baseline_tok = float(baseline_doc["tok_s"])
    baseline_ttft = float(baseline_doc.get("ttft_p95_ms") or 1152.24)
    ratio = tok_s / baseline_tok
    ttft_ratio = ttft_p95 / baseline_ttft
    compare = {
        "status": "PASS" if ratio >= 0.50 and success >= 0.99 and ttft_ratio <= 1.25 else "NO_GO",
        "source": "evalscope",
        "baseline_backend": "vllm",
        "candidate_backend": "infinilm",
        "success_rate_min": 0.99,
        "tok_s_floor": 0.50,
        "ttft_p95_max_ratio": 1.25,
        "tok_s_ratio": ratio,
        "ttft_p95_ratio": ttft_ratio,
        "baseline": {
            "tok_s": baseline_tok,
            "ttft_p95_ms": baseline_ttft,
            "success_rate": float(baseline_doc.get("success_rate", 1.0)),
        },
        "candidate": {
            "tok_s": tok_s,
            "ttft_p95_ms": ttft_p95,
            "success_rate": success,
        },
        "candidate_summary_ref": str(results[-1]),
        "measured_at": results[-1].parent.parent.parent.name,
        "note": "Fresh candidate-only P64/N64 wall against frozen vLLM baseline.",
    }
    target = SUITE / "compare.json"
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(compare, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(f"##autohelix[primary_wall_ratio={ratio:.8f}]")
    print(f"##autohelix[ttft_p95_ratio={ttft_ratio:.8f}]")
    print(f"##autohelix[success_rate={success:.8f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
