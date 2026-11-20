#!/usr/bin/env python3
"""Exit 0 iff unprofiled primary wall ratio > 1.0 (surpass). Human ships after."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_metrics import load_compare, metrics_from_compare, primary_surpassed, suite_dir  # noqa: E402


def main() -> int:
	suite = suite_dir()
	compare = load_compare(suite)
	if not compare:
		print(f"[stop_surpass] missing {suite / 'compare.json'}", file=sys.stderr)
		return 2
	try:
		metrics = metrics_from_compare(compare)
	except ValueError as e:
		print(f"[stop_surpass] {e}", file=sys.stderr)
		return 2
	ratio = metrics["primary_wall_ratio"]
	if primary_surpassed(metrics):
		print(f"[stop_surpass] SURPASSED primary_wall_ratio={ratio}", file=sys.stderr)
		print(
			"Human ship next: matrix refresh + ITW tag + IO case worktree bump.",
			file=sys.stderr,
		)
		return 0
	print(
		f"[stop_surpass] not surpassed primary_wall_ratio={ratio} (need > 1.0)",
		file=sys.stderr,
	)
	return 1


if __name__ == "__main__":
	raise SystemExit(main())
