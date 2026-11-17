# InfiniLM / vLLM performance investigation — 2026-10-28

This experiment uses an isolated snapshot of the supplied production pin in `worktree/`, mounted at the original container path to preserve build-cache paths. The production pin is not edited. The experiment container is `infinilm-perf-loop` on port 18000. The runner restores the original production container when the queue completes or is interrupted.

## Measurement audit

The supplied `profile_20261028-175202_fair` comparison cannot support its claimed fair kernel delta:

* MACA 3.8.2 emits epoch-nanosecond timestamps and nanosecond durations. The supplied parser assumes microseconds, inflating its printed milliseconds by 1000 and dropping valid long events at its duration cutoff.
* InfiniLM contains four GPUs. The recovered vLLM trace is truncated and contains only GPU 0; comparing their summed residency is invalid.
* InfiniLM's traced prefill has no text, no prompt token count, and no TTFT. Its reported one completion token is manufactured by the old workload's fallback; that is a failed request, not parity.
* The 95th percentile of HTOD event counts does not define a matched request window. The new runner records request boundaries in epoch nanoseconds and analyzes each GPU separately.
* The attempted attach-mode capture connected but produced an empty trace. It is rejected. All new inference capture uses launch-time mcTracer.

Clean supplied host reference: InfiniLM 6.473 tok/s, 28.638 s prefill TTFT; vLLM 42.585 tok/s, 1.990 s. A new unmodified InfiniLM repeat measured 6.66–6.77 tok/s and 28.765 s TTFT. These workloads are single-request TP4 Qwen3.8-27B on four C280 GPUs. The benchmark explicitly disables thinking, checks response completion/usage/content, and preserves text for correctness review.

## MCCL deep dive

Path: `BaseQuantization::forward_allreduce` → in-place `infinicore::op::distributed::allreduce_` → `infinicclAllReduce` → InfiniCCL `CclAllReduceImpl` → `McclApi::AllReduce` → vendor `mcclAllReduce`. There is no extra host copy or synchronization in the wrapper. Vendor MCCL is supplied as a binary, so internal implementation source is not available here; the investigation uses its public API, kernel names, topology/debug output, and measured behavior.

A standalone four-thread/four-rank BF16 correctness benchmark (`tools/mccl_bench.cpp`) checks every output element. Default FC reductions of 10 KiB took approximately 14–19 us per GPU including queued-loop overhead. Disabling FC cost about 27 us; forcing one FC block cost about 23–25 us. A 20 MiB reduction took about 367 us; these setting changes did not offer a compelling improvement. Launch-time tracing increased the microbenchmark loop overhead to about 72 us, so tracer-induced rank skew must not be equated with clean-server latency. The selected small-message kernel is `mcclKernel_AllReduce_FC4_LL_Sum_mccl_bfloat16`.

In the supplied InfiniLM trace's final eight-second exploratory window, ordinal alignment found 4,352 collectives on each GPU. Median rank-arrival skew is **467.968 us**, p95 **1226.496 us**; median time from last arrival to last completion is **7.424 us**. Of 7084.378 ms summed collective residency, 6951.649 ms occurs before the last rank arrives (**98.1%**). This is aggregate GPU residency, not 7 seconds of wall-clock communication. The main opportunity is removing rank launch skew and upstream stalls; replacing the already-fast FC4 kernel alone will not close the host gap. See `results/original_rank_skew.json`.

## Changes and experiments

1. Runtime environment defaults now permit candidate overrides in the isolated snapshot.
2. Piecewise prefill rebinds Mamba metadata to the updated device buffers. Without this, `to_model_input(cpu)` leaves `CausalConv1d` reading CPU metadata on replay. The first graph run demonstrated the error on all four ranks.
3. Delta-rule prefill keeps each thread's recurrent state column in registers, avoiding repeated full-state global-memory passes for every token. Tested against an independent FP32 recurrence and the original kernel, including indexed BF16 state pools, 64/128 head dimensions, and lengths 1/31/257/2048. Outputs matched the original in all tested cases. At length 2048, original 100.986 ms → candidate 20.253 ms (**4.99×**). At 257 tokens, 12.791 ms → 2.594 ms. The clean end-to-end register-state experiment measured **8.235 s TTFT**, down **71.4%** from the new 28.765 s baseline; decode remained **6.578 tok/s**. Its generated text still fails correctness. The initial traced restart exited during startup and was rejected (load-only traces are not compared).
4. A queued Qwen3.5 processor patch generates mRoPE positions for exactly each scheduled row's token slice. It fixes the observed 3×10718 → 3×2048 copy mismatch and handles mixed prefill/decode rows. Four CPU contract tests pass, including continuation-chunk Mamba state reuse (the old processor resets state on every prefill chunk).

The graph-only candidate was rejected: median short-request throughput was about 4.24 tok/s, and the long request failed with the position-shape mismatch. Further candidates use the validated register kernel and then the processor fix. Results are recorded in `results/ledger.json` and per-candidate directories.

The baseline itself generates malformed output for a simple repeat-`pad` prompt. The loop records this and refuses to classify a faster but incorrect candidate as passing. Neither HTTP 200 nor a nonempty string is treated as proof of model correctness. Later runs add arithmetic and capital-city smoke checks plus a fixed decode-length gate; the reference outputs are retained even when those checks fail.

## Remaining decode and trace limits

A new complete InfiniLM decode capture (`results/06_corrected_integrated_eager/traced/partial_decode_analysis.json`) again shows median rank skew **407.808 us** and last-arrival-to-finish **7.424 us**. Its two completed request windows have 1,792 collective kernels on **each** rank. The request emitted only 14 tokens (a refusal), so its throughput is not an accepted fixed-64-token comparison. The subsequent traced prefill failed and is explicitly excluded.

Decode graph diagnostics confirmed **17 device segments and 16 host segments** on each InfiniLM rank (1,075 recorded operators, 33 segments). MetaX is excluded by `MhaKVCache::supports_device_graph_capture`, so each full-attention layer remains a host break. Graph replay generated incorrect output and was rejected. Stabilizing MetaX graph replay and the attention provider's capture lifetime is a higher-priority decode direction than changing the already-fast MCCL FC4 kernel. Do not simply remove the capture guard: the linked Torch flash-attention provider allocates temporary results.

Fresh vLLM traces were collected from all four rank processes with no truncated JSON. They nevertheless lose GPU coverage during the second decode and contain no GPU events for the completed prefill request. They are **not a valid matched kernel comparison**. The new validity gate requires all four GPU windows and equal nonzero collective counts. The default full trace, a separate InfiniLM prefill-only attempt, and a lightweight kernel-only attempt all terminated the long request without text. A kernel-only configuration was verified with the standalone MCCL test, then tried on an isolated prefill capture. It reduced the inference trace to **37 MB**, but prefill still terminated without a response (`results/08_prefill_kernel_only/state.json`). This failure is not explained by JSON size alone. Default record-count flags were inspected rather than guessed: the SDK reports `TRACER_RECORD_NUM=0`, so no arbitrary finite record cap was installed.

The available source includes the MCCL wrapper, not vendor MCCL internals. New kernel traces, startup failures, candidate text, and the unsuccessful graph experiments are retained for follow-up. No production pin or release tag has been changed.

## Repeatable loop

`tools/loop.py` consumes `queue.json`: apply a checked local patch → rebuild → clean warmup/repeated benchmark → launch-time traced benchmark → validate trace → analyze request windows/rank skew → record verdict → next candidate. Source revision, diff, environment, outputs, build logs, trace, and analysis are retained. Startup-only trace failures have at most three attempts; each attempt has a separate directory and only the successful attempt is eligible for analysis. A failed source patch is reversed with `git apply -R`; no repository reset or production promotion occurs. New candidates can be added to the queue. `--watch` waits for new concrete candidates; it does not invent unreviewed source changes.

Run from this directory:

```bash
python3 tools/loop.py --restore-production
```

Optional `--watch` continues processing newly added candidates. Default proximity gates require correctness and valid capture, at least 80% of vLLM throughput, and at most 1.25× vLLM prefill TTFT. The queue now uses a fresh vLLM reference: **54.601 tok/s**, **2.059 s TTFT**, prefix caching disabled, identical prompt token counts (30 / 10,718). Raw output and accuracy limitations are in `results/vllm_current/REFERENCE_NOTES.md`. The new reference itself fails the arithmetic smoke; absolute model-accuracy claims are therefore blocked. Ctrl-C/SIGTERM stops the experiment and restores the production container. Each candidate's `state.json` shows the current phase and failures.

Tests: `python3 tools/test_profile.py`; `docker exec infinilm-perf-loop python /perf/tools/test_cgdr.py`; `docker exec infinilm-perf-loop python /perf/tools/test_positions.py`.

## Final state

Eight experiment/capture variants were exercised (six source/runtime candidates, then two isolated prefill captures). No candidate passed all acceptance gates. The repeatable loop, numerical tests, failed-run evidence, and cumulative source snapshot remain local. Patches in `patches/final/` separate the validated register-state kernel from graph metadata, loader/processor correctness fixes, diagnostics, and runtime override support.

Best clean prefill result: **8.235 s versus 28.765 s baseline** (71.4% less; ~3.49× faster). Fresh vLLM: **2.059 s**, so a substantial prefill gap remains. Decode has not converged: graph experiments peak around 7.75 tok/s with incorrect output, against the fresh vLLM 54.60 tok/s. Eager corrected candidates emit early refusals, so their reported ~5 tok/s is not a valid fixed-64-token speed comparison. Graph-mode and accuracy acceptance remain open; these results must not be presented as a production-ready optimization.

The original production container is restarted at the end; its verified health result is stored in `results/production_restored.json`. The production source/install tree and release tags were not modified. Experiment containers retain only idle entrypoints after cleanup.
