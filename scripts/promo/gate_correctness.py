#!/usr/bin/env python3
"""AutoHelix constraint: suite correctness / success_rate must pass."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/promo/gate_correctness.py` from ITW root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_metrics import correctness_ok, load_compare, metrics_from_compare, suite_dir  # noqa: E402


def main() -> int:
	suite = suite_dir()
	compare = load_compare(suite)
	if not compare:
		print(f"[gate_correctness] missing {suite / 'compare.json'}", file=sys.stderr)
		return 1
	try:
		metrics = metrics_from_compare(compare)
	except ValueError as e:
		print(f"[gate_correctness] {e}", file=sys.stderr)
		return 1
	if not correctness_ok(compare, metrics):
		print(
			f"[gate_correctness] FAIL success_rate={metrics['success_rate']} "
			f"status={compare.get('status')}",
			file=sys.stderr,
		)
		return 1
	print(
		f"[gate_correctness] PASS success_rate={metrics['success_rate']} "
		f"status={compare.get('status')}",
		file=sys.stderr,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
