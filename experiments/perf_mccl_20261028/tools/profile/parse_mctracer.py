"""Non-interactive Chrome-trace loader for MetaX mcTracer output.

Classifies events by pid matching /opt/maca-*/tools/cut_stat/pack_sql.py:
  pid 1              -> MACA API
  GPUList            -> GPU kernels
  DMAList            -> DMA / memcpy
  pid 5              -> TX
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .schema import HostStats, KernelStat, ProfileSummary

# Mirror pack_sql.py
GPU_PIDS = {2, 258, 514, 770, 1026, 1282, 1538, 1794}
DMA_PIDS = {3, 259, 515, 771, 1027, 1283, 1539, 1795}
API_PID = 1
TX_PID = 5

# Standard Chrome traces use microseconds; MACA epoch-ns traces are detected below.
US_TO_MS = 1e-3

FA_SPLITKV_RE = re.compile(r"flash_fwd_splitkv|splitkv_combine|flash_attn.*split", re.I)
MEMCPY_RE = re.compile(r"memcpy|dma|copy", re.I)
ALLREDUCE_RE = re.compile(r"AllReduce|mcclAllReduce", re.I)
HTOD_RE = re.compile(r"memcpy\s+HTOD", re.I)


def _iter_events(raw: Any) -> Iterable[Dict[str, Any]]:
	"""Yield Chrome complete events from common mcTracer JSON shapes."""
	if isinstance(raw, list):
		for item in raw:
			if isinstance(item, dict):
				yield item
		return
	if not isinstance(raw, dict):
		return
	# Standard chrome: {"traceEvents": [...]}
	events = raw.get("traceEvents")
	if isinstance(events, list):
		for item in events:
			if isinstance(item, dict):
				yield item
		return
	# pack_sql shape: top-level dict of list values
	for value in raw.values():
		if isinstance(value, list):
			for item in value:
				if isinstance(item, dict):
					yield item


def _dur_ms(item: Dict[str, Any], scale: float = US_TO_MS) -> Optional[float]:
	if "dur" not in item:
		return None
	try:
		ms = float(item["dur"]) * scale
	except (TypeError, ValueError):
		return None
	# Drop absurd durations from truncated/corrupt mcTracer records.
	if ms < 0 or ms > 60_000:
		return None
	return ms


def _accumulate(
	buckets: Dict[str, List[float]],
	name: str,
	ms: float,
) -> None:
	buckets[name].append(ms)


def _stream_trace_events(text: str) -> List[Dict[str, Any]]:
	"""Decode complete objects from a (possibly truncated) chrome traceEvents array."""
	decoder = json.JSONDecoder()
	i = text.find("[")
	if i < 0:
		return []
	i += 1
	n = len(text)
	events: List[Dict[str, Any]] = []
	while i < n:
		while i < n and text[i] in " \n\r\t,":
			i += 1
		if i >= n or text[i] == "]":
			break
		try:
			obj, end = decoder.raw_decode(text, i)
		except json.JSONDecodeError:
			break
		if isinstance(obj, dict):
			events.append(obj)
		i = end
	return events


def _load_chrome_json(path: Path) -> Any:
	"""Load Chrome JSON; stream-repair truncated mcTracer dumps."""
	text = path.read_text(encoding="utf-8-sig", errors="replace")
	try:
		return json.loads(text)
	except json.JSONDecodeError:
		events = _stream_trace_events(text)
		if not events:
			raise
		return {"traceEvents": events, "truncated": True}


def parse_mctracer(
	path: Union[str, Path],
	*,
	backend: str = "unknown",
	host: Optional[HostStats] = None,
	top_n: int = 20,
) -> ProfileSummary:
	"""Parse a mcTracer Chrome JSON file into ProfileSummary."""
	p = Path(path)
	# Accept UTF-8 BOM (PowerShell Set-Content often writes utf-8-sig).
	raw = _load_chrome_json(p)

	# MACA 3.8 emits epoch nanoseconds despite the Chrome JSON format.
	events = list(_iter_events(raw))
	scale = 1e-6 if any(int(e.get("ts", 0)) >= 100_000_000_000_000_000 for e in events) else US_TO_MS
	api_ms = 0.0
	gpu_ms = 0.0
	dma_ms = 0.0
	tx_ms = 0.0
	fa_splitkv_ms = 0.0
	allreduce_ms = 0.0
	htod_ms = 0.0
	kernel_durs: Dict[str, List[float]] = defaultdict(list)
	counts = {"truncated": bool(raw.get("truncated", False)) if isinstance(raw, dict) else False, "timestamp_unit_ns": scale == 1e-6, "api": 0, "gpu": 0, "dma": 0, "tx": 0, "metadata": 0, "other": 0}
	# (ts_us, ms) for post-load window split after weight-load HTOD cluster
	ar_events: List[Tuple[float, float]] = []
	htod_events: List[Tuple[float, float]] = []
	gpu_events: List[Tuple[float, float]] = []

	for item in events:
		if item.get("cat") == "__metadata":
			counts["metadata"] += 1
			continue
		pid = item.get("pid")
		name = str(item.get("name") or "")
		ms = _dur_ms(item, scale)
		if ms is None:
			counts["other"] += 1
			continue
		try:
			ts = float(item.get("ts") or 0.0)
		except (TypeError, ValueError):
			ts = 0.0

		if pid == API_PID:
			api_ms += ms
			counts["api"] += 1
		elif pid in GPU_PIDS:
			gpu_ms += ms
			counts["gpu"] += 1
			gpu_events.append((ts, ms))
			_accumulate(kernel_durs, name or "<unnamed>", ms)
			if FA_SPLITKV_RE.search(name):
				fa_splitkv_ms += ms
			if ALLREDUCE_RE.search(name):
				allreduce_ms += ms
				ar_events.append((ts, ms))
		elif pid in DMA_PIDS:
			dma_ms += ms
			counts["dma"] += 1
			if MEMCPY_RE.search(name):
				_accumulate(kernel_durs, name or "memcpy", ms)
			if HTOD_RE.search(name):
				htod_ms += ms
				htod_events.append((ts, ms))
		elif pid == TX_PID:
			tx_ms += ms
			counts["tx"] += 1
		else:
			counts["other"] += 1

	allreduce_ms_post_load = allreduce_ms
	htod_ms_post_load = htod_ms
	gpu_kernel_ms_post_load = gpu_ms
	if htod_events:
		htod_events.sort(key=lambda x: x[0])
		cut_i = min(len(htod_events) - 1, max(0, int(len(htod_events) * 0.95)))
		cut_ts = htod_events[cut_i][0]
		allreduce_ms_post_load = sum(ms for ts, ms in ar_events if ts >= cut_ts)
		htod_ms_post_load = sum(ms for ts, ms in htod_events if ts >= cut_ts)
		gpu_kernel_ms_post_load = sum(ms for ts, ms in gpu_events if ts >= cut_ts)
		counts["htod_cut_ts"] = int(cut_ts)
		counts["htod_pre_load_n"] = cut_i + 1
		counts["allreduce_post_load_n"] = sum(1 for ts, _ in ar_events if ts >= cut_ts)
		counts["gpu_post_load_n"] = sum(1 for ts, _ in gpu_events if ts >= cut_ts)

	top: List[KernelStat] = []
	ranked: List[Tuple[str, List[float]]] = sorted(
		kernel_durs.items(),
		key=lambda kv: sum(kv[1]),
		reverse=True,
	)
	for name, durs in ranked[: max(1, top_n)]:
		top.append(
			KernelStat(
				name=name,
				count=len(durs),
				total_ms=sum(durs),
				max_ms=max(durs) if durs else 0.0,
			)
		)

	return ProfileSummary(
		backend=backend,
		gpu_kernel_ms=gpu_ms,
		api_ms=api_ms,
		dma_ms=dma_ms,
		tx_ms=tx_ms,
		fa_splitkv_ms=fa_splitkv_ms,
		allreduce_ms=allreduce_ms,
		htod_ms=htod_ms,
		allreduce_ms_post_load=allreduce_ms_post_load,
		htod_ms_post_load=htod_ms_post_load,
		gpu_kernel_ms_post_load=gpu_kernel_ms_post_load,
		top_kernels=top,
		host=host or HostStats(),
		event_counts=counts,
		source=str(p),
	)


def _build_arg_parser() -> argparse.ArgumentParser:
	ap = argparse.ArgumentParser(
		prog="python -m profile.parse_mctracer",
		description="Parse mcTracer Chrome JSON into ProfileSummary JSON",
	)
	ap.add_argument("trace", type=Path, help="Path to tracer Chrome JSON")
	ap.add_argument("-o", "--output", type=Path, help="Write parsed JSON here")
	ap.add_argument("--backend", default="unknown", help="Backend label")
	ap.add_argument("--top-n", type=int, default=20, help="Top-N kernels")
	ap.add_argument("--tok-s", type=float, default=None, help="Optional host tok/s")
	ap.add_argument("--elapsed-s", type=float, default=None, help="Optional host elapsed seconds")
	ap.add_argument("--tokens", type=int, default=None, help="Optional host completion tokens")
	ap.add_argument("--workload", default=None, help="Optional workload name")
	ap.add_argument("--ttft-s", type=float, default=None, help="Optional host TTFT seconds")
	ap.add_argument("--prompt-tokens", type=int, default=None, help="Optional host prompt tokens")
	return ap


def main(argv: Optional[List[str]] = None) -> int:
	args = _build_arg_parser().parse_args(argv)
	host = HostStats(
		elapsed_s=args.elapsed_s,
		tokens=args.tokens,
		tok_s=args.tok_s,
		workload=args.workload,
		ttft_s=args.ttft_s,
		prompt_tokens=args.prompt_tokens,
	)
	summary = parse_mctracer(
		args.trace,
		backend=args.backend,
		host=host,
		top_n=args.top_n,
	)
	payload = json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n"
	if args.output:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(payload, encoding="utf-8")
		print(f"wrote {args.output}", flush=True)
	else:
		sys.stdout.write(payload)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
