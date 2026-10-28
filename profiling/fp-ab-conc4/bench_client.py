#!/usr/bin/env python3
"""Closed-loop concurrency benchmark for the InfiniLM chat API."""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def request_one(base_url: str, prompt: dict[str, Any], output_tokens: int, ignore_eos: bool = False, include_output_text: bool = False) -> dict[str, Any]:
    payload = {
        "model": "minicpm5-2b",
        "messages": [{"role": "user", "content": prompt["text"]}],
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "stream": True,
        "ignore_eos": ignore_eos,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    submitted = time.perf_counter()
    first_token = None
    output_chars = 0
    output_chunks = 0
    finish_reason = None
    text_chunks = []
    with urllib.request.urlopen(request, timeout=1200) as response:
        for raw_line in response:
            line = raw_line.strip()
            if not line.startswith(b"data: "):
                continue
            body = line[6:]
            if body == b"[DONE]":
                continue
            obj = json.loads(body)
            choice = obj.get("choices", [{}])[0]
            content = choice.get("content")
            if content is None:
                content = choice.get("delta", {}).get("content") or ""
            if content:
                if include_output_text:
                    text_chunks.append(content)
                if first_token is None:
                    first_token = time.perf_counter()
                output_chars += len(content)
                output_chunks += 1
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    completed = time.perf_counter()
    if finish_reason is None:
        raise RuntimeError(f"{prompt['id']}: stream ended without finish_reason")
    if first_token is None:
        raise RuntimeError(f"{prompt['id']}: stream ended without generated content")
    result = {
        "id": prompt["id"],
        "ttft_ms": (first_token - submitted) * 1000.0,
        "latency_ms": (completed - submitted) * 1000.0,
        "output_chunks": output_chunks,
        "output_chars": output_chars,
        "finish_reason": finish_reason,
    }
    if include_output_text:
        result["output_text"] = "".join(text_chunks)
    return result


def run_batch(base_url: str, prompts: list[dict[str, Any]], concurrency: int, output_tokens: int, ignore_eos: bool = False, include_output_text: bool = False) -> dict[str, Any]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(request_one, base_url, prompt, output_tokens, ignore_eos, include_output_text) for prompt in prompts]
        requests = [future.result() for future in futures]
    end = time.perf_counter()
    ttfts = [item["ttft_ms"] for item in requests]
    latencies = [item["latency_ms"] for item in requests]
    chunks = sum(item["output_chunks"] for item in requests)
    return {
        "makespan_s": end - start,
        "requests": len(requests),
        "output_chunks": chunks,
        "output_tokens_per_s": chunks / (end - start),
        "ttft_ms": {"min": min(ttfts), "p50": percentile(ttfts, 50), "max": max(ttfts)},
        "request_latency_ms": {
            "min": min(latencies),
            "p50": percentile(latencies, 50),
            "max": max(latencies),
        },
        "items": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["eager", "full_and_piecewise", "vllm_graph"], required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-only", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--include-output-text", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=0)
    args = parser.parse_args()

    data = json.loads(Path(args.prompts).read_text())
    prompts = data["prompts"]
    if args.max_prompts < 0:
        parser.error('--max-prompts must be nonnegative')
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    warmup = run_batch(args.base_url, prompts, args.concurrency, args.output_tokens, args.ignore_eos)
    result = {
        "mode": args.mode,
        "started_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workload": {
            "requests": len(prompts),
            "target_prompt_tokens": data["target_prompt_tokens"],
            "output_tokens": args.output_tokens,
            "concurrency": args.concurrency,
            "closed_loop": True,
            "ignore_eos": args.ignore_eos,
        },
        "warmup": warmup,
    }
    if not args.warmup_only:
        measured_start_wall = time.time()
        result["measured"] = run_batch(args.base_url, prompts, args.concurrency, args.output_tokens, args.ignore_eos, args.include_output_text)
        result["measured_wall"] = {
            "start_epoch": measured_start_wall,
            "end_epoch": time.time(),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    selected = warmup if args.warmup_only else result["measured"]
    print(json.dumps({
        "mode": args.mode,
        "warmup_only": args.warmup_only,
        "makespan_s": selected["makespan_s"],
        "output_chunks": selected["output_chunks"],
        "output_tokens_per_s": selected["output_tokens_per_s"],
        "p50_ttft_ms": selected["ttft_ms"]["p50"],
    }))


if __name__ == "__main__":
    main()
