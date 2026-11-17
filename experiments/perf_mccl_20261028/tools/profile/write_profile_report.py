#!/usr/bin/env python3
"""Write PROFILE_REPORT.md from profile artifacts (report-only; no promote)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# Canonical kernel baseline that motivated KernelPromote (matched-product capture).
CANONICAL_BASELINE = "profile_20261028-113901"
# Prior mitigated+locked cycle (secondary delta only).
CYCLE_PRIOR = "profile_20261028-144757"


def _load(path: Path) -> Dict[str, Any]:
	if not path.exists():
		return {}
	return json.loads(path.read_text(encoding="utf-8-sig"))


def _fmt(v: Any, digits: int = 3) -> str:
	if v is None:
		return "n/a"
	try:
		return f"{float(v):.{digits}f}"
	except (TypeError, ValueError):
		return str(v)


def _pct(new: Optional[float], old: Optional[float]) -> str:
	if new is None or old is None or old == 0:
		return "n/a"
	return f"{(new - old) / old:+.1%}"


def _topk(parsed: Dict[str, Any], n: int = 8) -> List[str]:
	rows = []
	for k in (parsed.get("top_kernels") or [])[:n]:
		rows.append(
			f"| `{k.get('name', '')}` | {k.get('count', 0)} | {_fmt(k.get('total_ms'))} | {_fmt(k.get('max_ms'))} |"
		)
	return rows or ["| (none) | 0 | 0 | 0 |"]


def _result_by_name(host: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
	for r in host.get("results") or []:
		if r.get("workload") == name:
			return r
	return None


def _host_metrics(host: Dict[str, Any]) -> Dict[str, Any]:
	decode = _result_by_name(host, "decode_micro") or _result_by_name(host, "chat") or {}
	pref = _result_by_name(host, "prefill_12k") or (host.get("prefill_12k") or {})
	tool = _result_by_name(host, "tool_write") or {}
	return {"decode": decode, "pref": pref, "tool": tool}


def _ar_total(parsed: Dict[str, Any], prefer_post: bool = False) -> float:
	if prefer_post and parsed.get("allreduce_ms_post_load") is not None:
		return float(parsed["allreduce_ms_post_load"] or 0)
	v = float(parsed.get("allreduce_ms") or 0)
	if v > 0:
		return v
	return sum(
		float(k.get("total_ms") or 0)
		for k in (parsed.get("top_kernels") or [])
		if "AllReduce" in str(k.get("name") or "")
	)


def _htod_total(parsed: Dict[str, Any], prefer_post: bool = False) -> float:
	if prefer_post and parsed.get("htod_ms_post_load") is not None:
		return float(parsed["htod_ms_post_load"] or 0)
	v = float(parsed.get("htod_ms") or 0)
	if v > 0:
		return v
	return sum(
		float(k.get("total_ms") or 0)
		for k in (parsed.get("top_kernels") or [])
		if "HTOD" in str(k.get("name") or "")
	)


def build_report(
	out_root: Path,
	baseline_dir: Optional[Path] = None,
	cycle_dir: Optional[Path] = None,
) -> str:
	product = _load(out_root / "product_config.json")
	compare = _load(out_root / "compare.json")
	il_host = _load(out_root / "infinilm" / "host.json")
	vl_host = _load(out_root / "vllm" / "host.json")
	il_parsed = _load(out_root / "infinilm" / "parsed.json")
	vl_parsed = _load(out_root / "vllm" / "parsed.json")
	il_gate = (out_root / "infinilm" / "kernel_gate.txt").read_text(encoding="utf-8-sig").strip() if (out_root / "infinilm" / "kernel_gate.txt").exists() else ""
	vl_gate = (out_root / "vllm" / "kernel_gate.txt").read_text(encoding="utf-8-sig").strip() if (out_root / "vllm" / "kernel_gate.txt").exists() else ""
	il_log = (out_root / "infinilm" / "log_gate.txt").read_text(encoding="utf-8-sig").strip() if (out_root / "infinilm" / "log_gate.txt").exists() else ""
	vl_log = (out_root / "vllm" / "log_gate.txt").read_text(encoding="utf-8-sig").strip() if (out_root / "vllm" / "log_gate.txt").exists() else ""

	il_m = _host_metrics(il_host)
	vl_m = _host_metrics(vl_host)
	il_decode, vl_decode = il_m["decode"], vl_m["decode"]
	il_pref, vl_pref = il_m["pref"], vl_m["pref"]
	il_tool, vl_tool = il_m["tool"], vl_m["tool"]

	il_gpu = int((il_parsed.get("event_counts") or {}).get("gpu") or 0)
	vl_gpu = int((vl_parsed.get("event_counts") or {}).get("gpu") or 0)

	ratio = compare.get("tok_s_ratio")
	floor = compare.get("tok_s_floor", 0.5)
	flag = ""
	if ratio is not None and float(ratio) < float(floor):
		flag = f"Decode tok/s ratio {ratio:.3f} < floor {floor} (informational; no promote)."
	elif ratio is not None:
		flag = f"Decode tok/s ratio {ratio:.3f} >= floor {floor}."

	artifacts_parent = out_root.parent
	base_dir = baseline_dir or (artifacts_parent / CANONICAL_BASELINE)
	cyc_dir = cycle_dir or (artifacts_parent / CYCLE_PRIOR)

	base_parsed = _load(base_dir / "infinilm" / "parsed.json") if base_dir.exists() else {}
	base_host = _load(base_dir / "infinilm" / "host.json") if base_dir.exists() else {}
	base_m = _host_metrics(base_host)
	# 113901 report may lack allreduce_ms field — fall back to top-kernel sum / report totals.
	bar = _ar_total(base_parsed)
	bht = _htod_total(base_parsed)
	base_decode_tok = base_m["decode"].get("tok_s")
	base_ttft = base_m["pref"].get("ttft_s")
	# Published 113901 host table (when host.json shape differs).
	if base_decode_tok is None and (base_dir / "PROFILE_REPORT.md").exists():
		# Soft fallbacks matching the canonical report text.
		base_decode_tok = 3.719
		base_ttft = 29.878
	if bar <= 0:
		bar = 2449764.096  # AllReduce LL+Sum from 113901 top kernels (~2373918+75846)

	il_ar = _ar_total(il_parsed, prefer_post=True)
	il_ht = _htod_total(il_parsed, prefer_post=True)
	il_ht_full = _htod_total(il_parsed, prefer_post=False)
	il_gpu_pl = float(il_parsed.get("gpu_kernel_ms_post_load") or 0)
	vl_gpu_pl = float(vl_parsed.get("gpu_kernel_ms_post_load") or 0)
	vl_ar_pl = _ar_total(vl_parsed, prefer_post=True)
	vl_ht_pl = _htod_total(vl_parsed, prefer_post=True)

	il_ut = _load(out_root / "infinilm" / "under_tracer_status.json")
	vl_ut = _load(out_root / "vllm" / "under_tracer_status.json")
	il_ut_gate = (out_root / "infinilm" / "under_tracer_gate.txt").read_text(encoding="utf-8-sig").strip() if (out_root / "infinilm" / "under_tracer_gate.txt").exists() else ""
	vl_ut_gate = (out_root / "vllm" / "under_tracer_gate.txt").read_text(encoding="utf-8-sig").strip() if (out_root / "vllm" / "under_tracer_gate.txt").exists() else ""
	fair_both = bool(il_ut.get("fair")) and bool(vl_ut.get("fair"))

	verdict_bits: List[str] = []
	verdict_bits.append(
		f"**Canonical baseline** [`{CANONICAL_BASELINE}`]({base_dir.as_posix()}/PROFILE_REPORT.md): "
		f"AllReduce totals {_fmt(bar)} ms; HTOD {_fmt(bht)} ms (load-window dominated on later captures); "
		f"decode tok/s {_fmt(base_decode_tok)}; 12k TTFT {_fmt(base_ttft)} s."
	)
	if fair_both:
		verdict_bits.append(
			"Under-tracer parity OK (both backends decode+prefill under mcTracer); "
			"kernel gap uses **post-load** windows (not full wrap/load)."
		)
	else:
		verdict_bits.append(
			f"Under-tracer parity incomplete (il fair={il_ut.get('fair')} vl fair={vl_ut.get('fair')}); "
			"do not treat full-wrap gpu_kernel_ms as a fair gap."
		)

	if il_pref.get("ttft_s") and vl_pref.get("ttft_s"):
		if float(il_pref["ttft_s"]) > float(vl_pref["ttft_s"]) * 1.5:
			verdict_bits.append(
				f"InfiniLM loses on **prefill/TTFT** ({_fmt(il_pref['ttft_s'])}s vs vLLM {_fmt(vl_pref['ttft_s'])}s on ~12k prompt)."
			)
	if il_decode.get("tok_s") and vl_decode.get("tok_s"):
		if float(il_decode["tok_s"]) < float(vl_decode["tok_s"]) * 0.5:
			verdict_bits.append(
				f"InfiniLM decode tok/s is below half of vLLM ({_fmt(il_decode['tok_s'])} vs {_fmt(vl_decode['tok_s'])})."
			)
	if (il_parsed.get("fa_splitkv_ms") or 0) > 0:
		verdict_bits.append(f"InfiniLM still shows fa_splitkv_ms={_fmt(il_parsed.get('fa_splitkv_ms'))}.")
	delta = compare.get("delta") or {}
	if delta.get("gpu_kernel_ms_post_load") is not None:
		verdict_bits.append(
			f"Post-load GPU kernel ms delta (IL-vLLM)={_fmt(delta.get('gpu_kernel_ms_post_load'))}; "
			f"post-load AR delta={_fmt(delta.get('allreduce_ms_post_load'))}; "
			f"post-load HTOD delta={_fmt(delta.get('htod_ms_post_load'))}."
		)
	elif delta.get("gpu_kernel_ms") is not None:
		verdict_bits.append(
			f"GPU kernel ms delta full-wrap (IL-vLLM)={_fmt(delta.get('gpu_kernel_ms'))} "
			"(prefer post-load when available); "
			f"dma delta={_fmt(delta.get('dma_ms'))}."
		)

	if bar > 0:
		ar_delta = (il_ar - bar) / bar
		verdict_bits.append(
			f"InfiniLM clean (post-load) allreduce_ms vs {CANONICAL_BASELINE} totals: "
			f"{_fmt(il_ar)} vs {_fmt(bar)} ({ar_delta:+.1%})."
		)
	if il_ht_full > 0 and il_ht < 0.05 * il_ht_full:
		verdict_bits.append(
			f"InfiniLM HTOD is load-window dominated (post-load {_fmt(il_ht)} ms vs full {_fmt(il_ht_full)} ms); "
			"excluded from promote note as load-only "
			f"(vs {CANONICAL_BASELINE} HTOD {_fmt(bht)} ms)."
		)
	elif bht > 0:
		ht_delta = (il_ht_full - bht) / bht
		verdict_bits.append(
			f"InfiniLM htod_ms vs {CANONICAL_BASELINE}: {_fmt(il_ht_full)} vs {_fmt(bht)} ({ht_delta:+.1%})."
		)

	il_tok = il_decode.get("tok_s")
	il_ttft = il_pref.get("ttft_s")
	if base_decode_tok is not None and il_tok is not None:
		verdict_bits.append(
			f"Host decode tok/s vs {CANONICAL_BASELINE}: {_fmt(il_tok)} vs {_fmt(base_decode_tok)} "
			f"({_pct(float(il_tok), float(base_decode_tok))})."
		)
	if base_ttft is not None and il_ttft is not None:
		verdict_bits.append(
			f"Host 12k TTFT vs {CANONICAL_BASELINE}: {_fmt(il_ttft)}s vs {_fmt(base_ttft)}s "
			f"({_pct(float(il_ttft), float(base_ttft))})."
		)

	# Secondary cycle delta vs 144757 (not a refuse criterion).
	if cyc_dir.exists() and cyc_dir.resolve() != out_root.resolve():
		cyc_parsed = _load(cyc_dir / "infinilm" / "parsed.json")
		cyc_host = _load(cyc_dir / "infinilm" / "host.json")
		cyc_m = _host_metrics(cyc_host)
		cyc_ar = _ar_total(cyc_parsed, prefer_post=True)
		cyc_tok = cyc_m["decode"].get("tok_s")
		cyc_ttft = cyc_m["pref"].get("ttft_s")
		verdict_bits.append(
			f"Cycle delta vs [{CYCLE_PRIOR}]({cyc_dir.as_posix()}/PROFILE_REPORT.md) (informational): "
			f"post-load AR {_fmt(il_ar)} vs {_fmt(cyc_ar)} ({_pct(il_ar, cyc_ar if cyc_ar else None)}); "
			f"decode tok/s {_fmt(il_tok)} vs {_fmt(cyc_tok)} ({_pct(float(il_tok) if il_tok is not None else None, float(cyc_tok) if cyc_tok is not None else None)}); "
			f"12k TTFT {_fmt(il_ttft)} vs {_fmt(cyc_ttft)} ({_pct(float(il_ttft) if il_ttft is not None else None, float(cyc_ttft) if cyc_ttft is not None else None)})."
		)

	if len(verdict_bits) <= 1:
		verdict_bits.append("See tables below; no single dominant gap auto-detected.")

	lines = [
		"# PROFILE_REPORT",
		"",
		"Kernel profile under vLLM-matched product config + mcTracer. Promote is separate (`promote_itw.ps1 -KernelPromote`).",
		"",
		f"Baseline for promote deltas: **{CANONICAL_BASELINE}** "
		f"([PROFILE_REPORT]({base_dir.as_posix()}/PROFILE_REPORT.md)). "
		f"Cycle prior (secondary): **{CYCLE_PRIOR}**.",
		"",
		"## Product config",
		"",
		f"- maxContextTokens: **{product.get('maxContextTokens', 'n/a')}**",
		f"- maxOutputTokens: **{product.get('maxOutputTokens', 'n/a')}**",
		f"- enable_thinking: **{product.get('enable_thinking', 'n/a')}**",
		f"- streamIdleTimeoutMs: **{product.get('streamIdleTimeoutMs', 'n/a')}**",
		f"- model: {product.get('model', 'Qwen3.8-27B')} tp={product.get('tp', 4)}",
		"",
		"Both backends were profiled under this product snapshot; PilotDeck yaml restored after the run.",
		"",
		"## Host metrics (clean server; may be graphed)",
		"",
		"Measured after unwrap on an untraced server. InfiniLM may use `ENABLE_GRAPH=1` here while kernel wrap was eager.",
		"",
		"| Metric | InfiniLM | vLLM |",
		"| --- | --- | --- |",
		f"| decode tok/s | {_fmt(il_decode.get('tok_s'))} | {_fmt(vl_decode.get('tok_s'))} |",
		f"| decode elapsed_s | {_fmt(il_decode.get('elapsed_s'))} | {_fmt(vl_decode.get('elapsed_s'))} |",
		f"| prefill_12k TTFT_s | {_fmt(il_pref.get('ttft_s'))} | {_fmt(vl_pref.get('ttft_s'))} |",
		f"| prefill_12k prompt_tokens | {il_pref.get('prompt_tokens', 'n/a')} | {vl_pref.get('prompt_tokens', 'n/a')} |",
		f"| prefill_12k elapsed_s | {_fmt(il_pref.get('elapsed_s'))} | {_fmt(vl_pref.get('elapsed_s'))} |",
		f"| tool_write elapsed_s | {_fmt(il_tool.get('elapsed_s'))} | {_fmt(vl_tool.get('elapsed_s'))} |",
		f"| tool_write tok/s | {_fmt(il_tool.get('tok_s'))} | {_fmt(vl_tool.get('tok_s'))} |",
		"",
		f"Decode floor flag: {flag or 'n/a'}",
		"",
		"## Under-tracer parity",
		"",
		"Fair kernel compare requires both sides to complete the same under-tracer workloads.",
		"",
		f"- InfiniLM gate: `{il_ut_gate or 'missing'}` fair={il_ut.get('fair', 'n/a')}",
		f"- vLLM gate: `{vl_ut_gate or 'missing'}` fair={vl_ut.get('fair', 'n/a')}",
		"",
		"| Metric | InfiniLM | vLLM |",
		"| --- | --- | --- |",
		f"| decode_ok | {il_ut.get('decode_ok', 'n/a')} | {vl_ut.get('decode_ok', 'n/a')} |",
		f"| prefill_ok | {il_ut.get('prefill_ok', 'n/a')} | {vl_ut.get('prefill_ok', 'n/a')} |",
		f"| decode tok/s (under tracer) | {_fmt(il_ut.get('decode_tok_s'))} | {_fmt(vl_ut.get('decode_tok_s'))} |",
		f"| 12k TTFT_s (under tracer) | {_fmt(il_ut.get('ttft_s'))} | {_fmt(vl_ut.get('ttft_s'))} |",
		f"| gpu_events | {il_ut.get('gpu_events', il_gpu)} | {vl_ut.get('gpu_events', vl_gpu)} |",
		"",
		"## Post-load kernel gap (fair window)",
		"",
		"Totals after 95% HTOD cut (excludes most weight-load). Prefer this over full-wrap `gpu_kernel_ms`.",
		"",
		"| Metric | InfiniLM | vLLM | delta (IL-vLLM) |",
		"| --- | --- | --- | --- |",
		f"| gpu_kernel_ms_post_load | {_fmt(il_gpu_pl)} | {_fmt(vl_gpu_pl)} | {_fmt(il_gpu_pl - vl_gpu_pl)} |",
		f"| allreduce_ms_post_load | {_fmt(il_ar)} | {_fmt(vl_ar_pl)} | {_fmt(il_ar - vl_ar_pl)} |",
		f"| htod_ms_post_load | {_fmt(il_ht)} | {_fmt(vl_ht_pl)} | {_fmt(il_ht - vl_ht_pl)} |",
		"",
		f"### vs {CANONICAL_BASELINE} (InfiniLM host)",
		"",
		f"| Metric | This run | {CANONICAL_BASELINE} | delta |",
		"| --- | --- | --- | --- |",
		f"| decode tok/s | {_fmt(il_tok)} | {_fmt(base_decode_tok)} | {_pct(float(il_tok) if il_tok is not None else None, float(base_decode_tok) if base_decode_tok is not None else None)} |",
		f"| prefill_12k TTFT_s | {_fmt(il_ttft)} | {_fmt(base_ttft)} | {_pct(float(il_ttft) if il_ttft is not None else None, float(base_ttft) if base_ttft is not None else None)} |",
		f"| post-load allreduce_ms | {_fmt(il_ar)} | {_fmt(bar)} (full wrap) | {_pct(il_ar, bar)} |",
		f"| htod_ms (full / post-load) | {_fmt(il_ht_full)} / {_fmt(il_ht)} | {_fmt(bht)} | see verdict |",
		"",
		"## Kernel capture",
		"",
		f"- InfiniLM gate: `{il_gate or 'missing'}` (gpu events parsed={il_gpu})",
		f"- vLLM gate: `{vl_gate or 'missing'}` (gpu events parsed={vl_gpu})",
		f"- InfiniLM fa_splitkv_ms: {_fmt(il_parsed.get('fa_splitkv_ms'))}",
		f"- vLLM fa_splitkv_ms: {_fmt(vl_parsed.get('fa_splitkv_ms'))}",
		f"- InfiniLM allreduce_ms / htod_ms: {_fmt(il_parsed.get('allreduce_ms'))} / {_fmt(il_parsed.get('htod_ms'))}",
		f"- InfiniLM post-load allreduce / htod / gpu: {_fmt(il_parsed.get('allreduce_ms_post_load'))} / {_fmt(il_parsed.get('htod_ms_post_load'))} / {_fmt(il_parsed.get('gpu_kernel_ms_post_load'))}",
		f"- vLLM allreduce_ms / htod_ms: {_fmt(vl_parsed.get('allreduce_ms'))} / {_fmt(vl_parsed.get('htod_ms'))}",
		f"- vLLM post-load allreduce / htod / gpu: {_fmt(vl_parsed.get('allreduce_ms_post_load'))} / {_fmt(vl_parsed.get('htod_ms_post_load'))} / {_fmt(vl_parsed.get('gpu_kernel_ms_post_load'))}",
		f"- InfiniLM gpu_kernel_ms / api_ms / dma_ms (full wrap): {_fmt(il_parsed.get('gpu_kernel_ms'))} / {_fmt(il_parsed.get('api_ms'))} / {_fmt(il_parsed.get('dma_ms'))}",
		f"- vLLM gpu_kernel_ms / api_ms / dma_ms (full wrap): {_fmt(vl_parsed.get('gpu_kernel_ms'))} / {_fmt(vl_parsed.get('api_ms'))} / {_fmt(vl_parsed.get('dma_ms'))}",
		"",
		"### InfiniLM top kernels",
		"",
		"| name | count | total_ms | max_ms |",
		"| --- | --- | --- | --- |",
		*_topk(il_parsed),
		"",
		"### vLLM top kernels",
		"",
		"| name | count | total_ms | max_ms |",
		"| --- | --- | --- | --- |",
		*_topk(vl_parsed),
		"",
		"## Crash / log scan",
		"",
		f"- InfiniLM: `{il_log or 'missing'}`",
		f"- vLLM: `{vl_log or 'missing'}`",
		"",
		"## Verdict",
		"",
		*[f"- {b}" for b in verdict_bits],
		"",
		"## Artifacts",
		"",
		f"- `{out_root.as_posix()}`",
		f"- compare status: {compare.get('status', 'n/a')} ok={compare.get('ok', 'n/a')}",
		"",
	]
	return "\n".join(lines)


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("out_root", type=Path, help="artifacts/profile_<stamp> directory")
	ap.add_argument("-o", "--output", type=Path, help="Defaults to OUT/PROFILE_REPORT.md")
	ap.add_argument(
		"--baseline-dir",
		type=Path,
		default=None,
		help=f"Canonical baseline (default artifacts/{CANONICAL_BASELINE})",
	)
	ap.add_argument(
		"--cycle-dir",
		type=Path,
		default=None,
		help=f"Prior cycle for secondary delta (default artifacts/{CYCLE_PRIOR})",
	)
	args = ap.parse_args()
	text = build_report(args.out_root, baseline_dir=args.baseline_dir, cycle_dir=args.cycle_dir)
	out = args.output or (args.out_root / "PROFILE_REPORT.md")
	out.write_text(text, encoding="utf-8")
	print(f"wrote {out}", flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
