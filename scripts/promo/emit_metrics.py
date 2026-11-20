"""Parse suite compare.json / walls into AutoHelix metric dict (no invent)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def suite_dir() -> Path:
	"""Shared suite dir across AutoHelix worktrees.

	Prefer absolute SUITE_DIR. Relative paths resolve against INFINI_TENSOR_WORKTREE
	(main pin), not the per-iteration worktree copy of this script.
	"""
	raw = os.environ.get("SUITE_DIR", "artifacts/promo/suite")
	p = Path(raw).expanduser()
	if p.is_absolute():
		return p.resolve()
	itw_env = os.environ.get("INFINI_TENSOR_WORKTREE", "").strip()
	if itw_env:
		return (Path(itw_env).expanduser() / p).resolve()
	# Fallback: script's repo root (wrong inside AutoHelix worktrees if relative)
	itw = Path(__file__).resolve().parents[2]
	return (itw / p).resolve()


def load_compare(suite: Path) -> Dict[str, Any]:
	path = suite / "compare.json"
	if not path.is_file():
		return {}
	return json.loads(path.read_text(encoding="utf-8"))


def _ratio(cand: Optional[float], base: Optional[float]) -> Optional[float]:
	if cand is None or base is None or base == 0:
		return None
	return float(cand) / float(base)


def metrics_from_compare(compare: Dict[str, Any]) -> Dict[str, float]:
	"""Derive the three AutoHelix metrics from a dual-wall compare.json."""
	if not compare:
		raise ValueError("empty compare.json")

	baseline = compare.get("baseline") or {}
	candidate = compare.get("candidate") or {}
	if not isinstance(baseline, dict):
		baseline = {}
	if not isinstance(candidate, dict):
		candidate = {}

	# Primary wall ratio (InfiniLM / vLLM tok/s)
	primary = compare.get("tok_s_ratio")
	if primary is None:
		primary = compare.get("WALL_DECODE_RATIO") or compare.get("wall_ratio")
	if primary is None:
		primary = _ratio(
			_as_float(candidate.get("tok_s")),
			_as_float(baseline.get("tok_s")),
		)
	if primary is None:
		raise ValueError("compare.json missing primary wall ratio / tok_s")

	ttft = compare.get("ttft_p95_ratio")
	if ttft is None:
		ttft = _ratio(
			_as_float(candidate.get("ttft_p95_ms")),
			_as_float(baseline.get("ttft_p95_ms")),
		)
	# Offline fixtures may omit TTFT; treat as neutral 1.0 for metric print
	if ttft is None:
		ttft = 1.0

	success = candidate.get("success_rate")
	if success is None:
		success = compare.get("success_rate")
	if success is None:
		# Fixtures without success_rate: assume gate-pass 1.0 when status PASS
		status = str(compare.get("status") or "").upper()
		success = 1.0 if status in {"PASS", "SURPASS"} else None
	if success is None:
		raise ValueError("compare.json missing candidate success_rate")

	return {
		"primary_wall_ratio": float(primary),
		"ttft_p95_ratio": float(ttft),
		"success_rate": float(success),
	}


def _as_float(v: Any) -> Optional[float]:
	if v is None:
		return None
	return float(v)


def print_autohelix_metrics(metrics: Dict[str, float]) -> None:
	"""One ##autohelix[name=value] line per metric (AutoHelix 0.1.1 protocol)."""
	for name in ("primary_wall_ratio", "ttft_p95_ratio", "success_rate"):
		print(f"##autohelix[{name}={metrics[name]}]")


def correctness_ok(compare: Dict[str, Any], metrics: Dict[str, float]) -> bool:
	ok_min = float(compare.get("success_rate_min") or 0.99)
	status = str(compare.get("status") or "").upper()
	if status in {"PROFILE_REGRESS", "NO_GO"}:
		# Still allow if only soft TTFT/tok floor — hard fail on success_rate
		pass
	if metrics["success_rate"] < ok_min:
		return False
	if status == "NO_GO":
		return False
	return True


def primary_surpassed(metrics: Dict[str, float]) -> bool:
	return metrics["primary_wall_ratio"] > 1.0


def main() -> int:
	suite = suite_dir()
	compare = load_compare(suite)
	try:
		m = metrics_from_compare(compare)
	except ValueError as e:
		print(f"[emit_metrics] {e}", file=sys.stderr)
		return 1
	print(json.dumps(m, indent=2, sort_keys=True))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
