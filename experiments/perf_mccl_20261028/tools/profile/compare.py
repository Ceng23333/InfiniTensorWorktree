"""Diff two ProfileSummary objects with tok/s and FA regression gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import ProfileSummary


DEFAULT_TOK_S_FLOOR = 0.50  # InfiniLM must be >= 50% of vLLM on same workload


def compare_summaries(
	infinilm: ProfileSummary,
	vllm: ProfileSummary,
	*,
	tok_s_floor: float = DEFAULT_TOK_S_FLOOR,
	fa_splitkv_ms_max: float = 0.0,
) -> Dict[str, Any]:
	"""Return compare.json payload. ok=true when gates pass."""
	reasons: List[str] = []

	il_tok = infinilm.host.tok_s
	vl_tok = vllm.host.tok_s
	ratio: Optional[float] = None
	if il_tok is None or vl_tok is None:
		reasons.append("missing host.tok_s on one or both summaries")
	elif vl_tok <= 0:
		reasons.append(f"vllm tok_s non-positive: {vl_tok}")
	else:
		ratio = float(il_tok) / float(vl_tok)
		if ratio < tok_s_floor:
			reasons.append(
				f"PROFILE_REGRESS tok_s ratio {ratio:.3f} < floor {tok_s_floor:.3f} "
				f"(infinilm={il_tok:.3f} vllm={vl_tok:.3f})"
			)

	if infinilm.fa_splitkv_ms > fa_splitkv_ms_max:
		reasons.append(
			f"infinilm fa_splitkv_ms={infinilm.fa_splitkv_ms:.3f} exceeds max {fa_splitkv_ms_max:.3f}"
		)

	ok = len(reasons) == 0
	# Fair kernel compare uses post-load windows (after 95% HTOD cut), not full wrap.
	il_gpu_pl = float(infinilm.gpu_kernel_ms_post_load or 0.0)
	vl_gpu_pl = float(vllm.gpu_kernel_ms_post_load or 0.0)
	il_ar_pl = float(infinilm.allreduce_ms_post_load or 0.0)
	vl_ar_pl = float(vllm.allreduce_ms_post_load or 0.0)
	il_ht_pl = float(infinilm.htod_ms_post_load or 0.0)
	vl_ht_pl = float(vllm.htod_ms_post_load or 0.0)
	return {
		"ok": ok,
		"status": "PASS" if ok else "PROFILE_REGRESS",
		"tok_s_floor": tok_s_floor,
		"tok_s_ratio": ratio,
		"infinilm": {
			"tok_s": il_tok,
			"gpu_kernel_ms": infinilm.gpu_kernel_ms,
			"gpu_kernel_ms_post_load": il_gpu_pl,
			"fa_splitkv_ms": infinilm.fa_splitkv_ms,
			"api_ms": infinilm.api_ms,
			"dma_ms": infinilm.dma_ms,
			"allreduce_ms": infinilm.allreduce_ms,
			"allreduce_ms_post_load": il_ar_pl,
			"htod_ms": infinilm.htod_ms,
			"htod_ms_post_load": il_ht_pl,
		},
		"vllm": {
			"tok_s": vl_tok,
			"gpu_kernel_ms": vllm.gpu_kernel_ms,
			"gpu_kernel_ms_post_load": vl_gpu_pl,
			"fa_splitkv_ms": vllm.fa_splitkv_ms,
			"api_ms": vllm.api_ms,
			"dma_ms": vllm.dma_ms,
			"allreduce_ms": vllm.allreduce_ms,
			"allreduce_ms_post_load": vl_ar_pl,
			"htod_ms": vllm.htod_ms,
			"htod_ms_post_load": vl_ht_pl,
		},
		"delta": {
			# Full-wrap deltas kept for legacy; prefer post_load_* for fair notes.
			"gpu_kernel_ms": infinilm.gpu_kernel_ms - vllm.gpu_kernel_ms,
			"api_ms": infinilm.api_ms - vllm.api_ms,
			"dma_ms": infinilm.dma_ms - vllm.dma_ms,
			"fa_splitkv_ms": infinilm.fa_splitkv_ms - vllm.fa_splitkv_ms,
			"allreduce_ms": infinilm.allreduce_ms - vllm.allreduce_ms,
			"htod_ms": infinilm.htod_ms - vllm.htod_ms,
			"gpu_kernel_ms_post_load": il_gpu_pl - vl_gpu_pl,
			"allreduce_ms_post_load": il_ar_pl - vl_ar_pl,
			"htod_ms_post_load": il_ht_pl - vl_ht_pl,
		},
		"reasons": reasons,
	}


def _load_summary(path: Path) -> ProfileSummary:
	return ProfileSummary.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))


def main(argv: Optional[List[str]] = None) -> int:
	ap = argparse.ArgumentParser(description="Compare InfiniLM vs vLLM profile summaries")
	ap.add_argument("infinilm", type=Path, help="parsed.json for InfiniLM")
	ap.add_argument("vllm", type=Path, help="parsed.json for vLLM")
	ap.add_argument("-o", "--output", type=Path, help="Write compare.json")
	ap.add_argument("--tok-s-floor", type=float, default=DEFAULT_TOK_S_FLOOR)
	ap.add_argument("--fa-splitkv-ms-max", type=float, default=0.0)
	args = ap.parse_args(argv)

	result = compare_summaries(
		_load_summary(args.infinilm),
		_load_summary(args.vllm),
		tok_s_floor=args.tok_s_floor,
		fa_splitkv_ms_max=args.fa_splitkv_ms_max,
	)
	payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
	if args.output:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(payload, encoding="utf-8")
		print(f"wrote {args.output}", flush=True)
	else:
		sys.stdout.write(payload)
	return 0 if result["ok"] else 2


if __name__ == "__main__":
	raise SystemExit(main())
