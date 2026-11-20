#!/usr/bin/env python3
"""AutoHelix metrics command: deterministic Observe→Parse via InfiniOrchestrator.

Prints one ##autohelix[name=value] line per metric for AutoHelix 0.1.1.
Does not apply levers; AutoHelix + Codex own iteration.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emit_metrics import (  # noqa: E402
	load_compare,
	metrics_from_compare,
	print_autohelix_metrics,
	suite_dir,
)

CASE_VLLM = "playground/Standalone/qwen3-8b-mars-vllm--unified-tp4"
CASE_INFINILM = "playground/Standalone/qwen3-8b-mars-inf--unified-tp4"


def _truthy(name: str, default: str = "0") -> bool:
	return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _io_root() -> Path:
	raw = os.environ.get("INFINI_ORCHESTRATOR", "").strip()
	if raw:
		return Path(raw).expanduser().resolve()
	# Sibling layout: deploy20261113/{InfiniTensorWorktree,InfiniOrchestrator}
	# Prefer workspace live IO next to deploy sibling if present.
	itw = Path(__file__).resolve().parents[2]
	candidates = [
		itw.parent / "InfiniOrchestrator",
		Path("/root/zenghua/workspace/InfiniOrchestrator"),
		itw.parent.parent / "InfiniOrchestrator",
	]
	for c in candidates:
		if (c / "harness" / "bin" / "run-promotion-loop-step").is_file():
			return c.resolve()
	raise SystemExit(
		"[observe] set INFINI_ORCHESTRATOR to InfiniOrchestrator root "
		"(need harness/bin/run-promotion-loop-step)"
	)


def _run_loop_step(io: Path, suite: Path) -> int:
	bin_path = io / "harness" / "bin" / "run-promotion-loop-step"
	env = os.environ.copy()
	env.setdefault("DEVICE_TRACK", os.environ.get("DEVICE_TRACK", "mars"))
	env.setdefault("HOST_ROLE", os.environ.get("HOST_ROLE", "mars"))
	env.setdefault("HOST_ID", os.environ.get("HOST_ID", "metax-9"))
	env.setdefault("MODEL", os.environ.get("MODEL", "Qwen3-8B"))
	env["SUITE_DIR"] = str(suite)
	# PYTHONPATH for adapters package
	harness = str(io / "harness")
	env["PYTHONPATH"] = harness + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

	cmd = [
		str(bin_path),
		"--suite-dir",
		str(suite),
		"--device-track",
		env["DEVICE_TRACK"],
		"--host",
		env["HOST_ID"],
		"--case-vllm",
		CASE_VLLM,
		"--case-infinilm",
		CASE_INFINILM,
		"--model",
		env["MODEL"],
	]
	if _truthy("RUN_WALL"):
		cmd.append("--run-wall")
		if os.environ.get("BASE_URL_VLLM"):
			cmd += ["--base-url-vllm", os.environ["BASE_URL_VLLM"]]
		if os.environ.get("BASE_URL_INFINILM"):
			cmd += ["--base-url-infinilm", os.environ["BASE_URL_INFINILM"]]
	else:
		cmd.append("--reuse-wall")

	if _truthy("SKIP_PROFILER", "1" if not os.environ.get("HCTRACER_ROOT") else "0"):
		cmd.append("--skip-profiler")
	elif _truthy("FORCE_PROFILER"):
		cmd.append("--force-profiler")

	print(f"[observe] exec: {' '.join(cmd)}", file=sys.stderr)
	proc = subprocess.run(cmd, cwd=str(io / "harness"), env=env)
	return int(proc.returncode)


def main() -> int:
	suite = suite_dir()
	suite.mkdir(parents=True, exist_ok=True)

	# Offline path: staged compare.json + RUN_WALL=0
	if not _truthy("RUN_WALL"):
		compare = load_compare(suite)
		if not compare:
			print(
				f"[observe] RUN_WALL=0 but missing {suite / 'compare.json'}",
				file=sys.stderr,
			)
			return 2
		# Still invoke loop_step for facts/backlog when IO is available
		try:
			io = _io_root()
		except SystemExit as e:
			print(str(e), file=sys.stderr)
			# Emit metrics from staged compare alone (dry fixture mode)
			try:
				metrics = metrics_from_compare(compare)
			except ValueError as ve:
				print(f"[observe] {ve}", file=sys.stderr)
				return 1
			print_autohelix_metrics(metrics)
			return 0
		rc = _run_loop_step(io, suite)
		if rc != 0:
			# Fall back to emit from existing compare if loop_step failed soft
			print(f"[observe] loop_step rc={rc}; emitting from staged compare", file=sys.stderr)
		try:
			metrics = metrics_from_compare(load_compare(suite))
		except ValueError as ve:
			print(f"[observe] {ve}", file=sys.stderr)
			return 1
		print_autohelix_metrics(metrics)
		return 0 if rc == 0 else rc

	# Live wall
	io = _io_root()
	rc = _run_loop_step(io, suite)
	if rc != 0:
		print(f"[observe] loop_step failed rc={rc}", file=sys.stderr)
		return rc
	try:
		metrics = metrics_from_compare(load_compare(suite))
	except ValueError as ve:
		print(f"[observe] {ve}", file=sys.stderr)
		return 1
	print_autohelix_metrics(metrics)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
