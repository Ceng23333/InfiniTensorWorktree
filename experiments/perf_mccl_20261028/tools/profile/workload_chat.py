#!/usr/bin/env python3
"""Prod-shaped workloads for profile/run_profile.ps1 (decode / prefill / tool Write)."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 600.0) -> Dict[str, Any]:
	data = json.dumps(payload).encode("utf-8")
	req = urllib.request.Request(
		url,
		data=data,
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	with urllib.request.urlopen(req, timeout=timeout) as resp:
		return json.loads(resp.read().decode("utf-8"))


def _stream_chat(
	url: str,
	payload: Dict[str, Any],
	timeout: float = 900.0,
) -> Tuple[str, Optional[int], Optional[int], Optional[float]]:
	"""Return text, completion_tokens, prompt_tokens, ttft_s."""
	data = json.dumps(payload).encode("utf-8")
	req = urllib.request.Request(
		url,
		data=data,
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	text_parts: List[str] = []
	completion_tokens: Optional[int] = None
	prompt_tokens: Optional[int] = None
	ttft_s: Optional[float] = None
	t0 = time.perf_counter()
	with urllib.request.urlopen(req, timeout=timeout) as resp:
		for raw in resp:
			line = raw.decode("utf-8", errors="replace").strip()
			if not line.startswith("data:"):
				continue
			body = line[5:].strip()
			if body == "[DONE]":
				break
			try:
				chunk = json.loads(body)
			except json.JSONDecodeError:
				continue
			usage = chunk.get("usage") or {}
			if usage.get("completion_tokens") is not None:
				completion_tokens = int(usage["completion_tokens"])
			if usage.get("prompt_tokens") is not None:
				prompt_tokens = int(usage["prompt_tokens"])
			for choice in chunk.get("choices") or []:
				delta = choice.get("delta") or {}
				piece = ""
				if delta.get("content"):
					piece = str(delta["content"])
					text_parts.append(piece)
				for tc in delta.get("tool_calls") or []:
					fn = tc.get("function") or {}
					if fn.get("arguments"):
						piece = str(fn["arguments"])
						text_parts.append(piece)
				if piece and ttft_s is None:
					ttft_s = time.perf_counter() - t0
	return "".join(text_parts), completion_tokens, prompt_tokens, ttft_s


def _pad_prompt(target_chars: int) -> str:
	unit = "The quick brown fox jumps over the lazy dog. "
	reps = max(1, (target_chars // len(unit)) + 1)
	body = (unit * reps)[:target_chars]
	return (
		"You are a coding agent. Ignore the filler below except as context length.\n"
		"At the end, reply with exactly one short sentence confirming you read the prompt.\n\n"
		f"{body}\n\n"
		"Confirm with one short sentence."
	)


def run_decode_micro(base: str, model: str, max_tokens: int = 64) -> Dict[str, Any]:
	"""Short prompt, decode-bound tok/s (non-stream)."""
	url = base.rstrip("/") + "/v1/chat/completions"
	payload = {
		"model": model,
		"temperature": 0,
		"max_tokens": max_tokens,
		"stream": False,
		"messages": [
			{
				"role": "user",
				"content": (
					f"Output the word pad exactly {max_tokens} times, "
					"separated by spaces. No other text."
				),
			}
		],
	}
	t0 = time.perf_counter()
	resp = _post_json(url, payload)
	elapsed = time.perf_counter() - t0
	usage = resp.get("usage") or {}
	tokens = int(usage.get("completion_tokens") or 0)
	prompt_tokens = int(usage.get("prompt_tokens") or 0)
	if tokens <= 0:
		content = (((resp.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
		tokens = max(1, len(content) // 4)
	tok_s = tokens / elapsed if elapsed > 0 else 0.0
	return {
		"workload": "decode_micro",
		"elapsed_s": elapsed,
		"tokens": tokens,
		"completion_tokens": tokens,
		"prompt_tokens": prompt_tokens,
		"ttft_s": None,
		"tok_s": tok_s,
		"finish_reason": ((resp.get("choices") or [{}])[0].get("finish_reason")),
	}


def run_prefill_stress(
	base: str,
	model: str,
	*,
	approx_prompt_tokens: int = 12000,
	max_tokens: int = 64,
) -> Dict[str, Any]:
	"""~12k-token prompt (prod agent-sized), stream for TTFT."""
	url = base.rstrip("/") + "/v1/chat/completions"
	# ~4 chars/token heuristic for English filler
	prompt = _pad_prompt(approx_prompt_tokens * 4)
	payload = {
		"model": model,
		"temperature": 0,
		"max_tokens": max_tokens,
		"stream": True,
		"stream_options": {"include_usage": True},
		"messages": [{"role": "user", "content": prompt}],
	}
	t0 = time.perf_counter()
	text, tokens, prompt_tokens, ttft_s = _stream_chat(url, payload, timeout=900.0)
	elapsed = time.perf_counter() - t0
	if not tokens or tokens <= 0:
		tokens = max(1, len(text) // 4)
	tok_s = tokens / elapsed if elapsed > 0 else 0.0
	return {
		"workload": "prefill_12k",
		"elapsed_s": elapsed,
		"tokens": tokens,
		"completion_tokens": tokens,
		"prompt_tokens": prompt_tokens,
		"ttft_s": ttft_s,
		"tok_s": tok_s,
		"text_len": len(text),
		"approx_prompt_chars": len(prompt),
	}


def run_tool_write(base: str, model: str, max_tokens: int = 128) -> Dict[str, Any]:
	url = base.rstrip("/") + "/v1/chat/completions"
	tools = [
		{
			"type": "function",
			"function": {
				"name": "Write",
				"description": "Write a file",
				"parameters": {
					"type": "object",
					"properties": {
						"path": {"type": "string"},
						"contents": {"type": "string"},
					},
					"required": ["path", "contents"],
				},
			},
		}
	]
	payload = {
		"model": model,
		"temperature": 0,
		"max_tokens": max_tokens,
		"stream": True,
		"stream_options": {"include_usage": True},
		"tools": tools,
		"tool_choice": "auto",
		"messages": [
			{
				"role": "user",
				"content": "Call the Write tool once to create outline.md with a one-line title.",
			}
		],
	}
	t0 = time.perf_counter()
	text, tokens, prompt_tokens, ttft_s = _stream_chat(url, payload)
	elapsed = time.perf_counter() - t0
	if not tokens or tokens <= 0:
		tokens = max(1, len(text) // 4)
	tok_s = tokens / elapsed if elapsed > 0 else 0.0
	return {
		"workload": "tool_write",
		"elapsed_s": elapsed,
		"tokens": tokens,
		"completion_tokens": tokens,
		"prompt_tokens": prompt_tokens,
		"ttft_s": ttft_s,
		"tok_s": tok_s,
		"text_len": len(text),
	}


# Back-compat alias
def run_chat(base: str, model: str, max_tokens: int = 64) -> Dict[str, Any]:
	r = run_decode_micro(base, model, max_tokens=max_tokens)
	r["workload"] = "chat"
	return r


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--base", default="http://127.0.0.1:8000")
	ap.add_argument("--model", default="Qwen3.8-27B")
	ap.add_argument(
		"--workload",
		choices=("chat", "decode_micro", "prefill_12k", "tool_write", "both", "prod"),
		default="prod",
	)
	ap.add_argument("--approx-prompt-tokens", type=int, default=12000)
	ap.add_argument("-o", "--output", help="Write JSON result path")
	args = ap.parse_args()

	results: List[Dict[str, Any]] = []
	try:
		# prod order: short GPU-active workloads first so mcTracer stays busy /
		# flushes useful kernels before the long 12k prefill (buffer pressure).
		if args.workload in ("chat", "decode_micro", "both", "prod"):
			results.append(run_decode_micro(args.base, args.model))
		if args.workload in ("tool_write", "both", "prod"):
			results.append(run_tool_write(args.base, args.model))
		if args.workload in ("prefill_12k", "prod"):
			results.append(
				run_prefill_stress(
					args.base,
					args.model,
					approx_prompt_tokens=args.approx_prompt_tokens,
				)
			)
	except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
		print(json.dumps({"ok": False, "error": str(e)}), flush=True)
		return 1

	# Primary host metric: decode_micro when present, else first result.
	primary = next((r for r in results if r["workload"] in ("decode_micro", "chat")), results[0])
	prefill = next((r for r in results if r["workload"] == "prefill_12k"), None)
	out: Dict[str, Any] = {
		"ok": True,
		"results": results,
		"elapsed_s": primary["elapsed_s"],
		"tokens": primary["tokens"],
		"tok_s": primary["tok_s"],
		"workload": primary["workload"],
		"prompt_tokens": primary.get("prompt_tokens"),
		"ttft_s": primary.get("ttft_s"),
	}
	if prefill:
		out["prefill_12k"] = {
			"elapsed_s": prefill["elapsed_s"],
			"tok_s": prefill["tok_s"],
			"ttft_s": prefill.get("ttft_s"),
			"prompt_tokens": prefill.get("prompt_tokens"),
			"completion_tokens": prefill.get("completion_tokens"),
		}
	payload = json.dumps(out, indent=2, sort_keys=True) + "\n"
	if args.output:
		with open(args.output, "w", encoding="utf-8") as f:
			f.write(payload)
	sys.stdout.write(payload)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
