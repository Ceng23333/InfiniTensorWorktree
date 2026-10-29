# Output throughput gap: InfiniLM versus vLLM

Measured artifacts: 2026-10-28; analysis: 2026-10-29. TP4 BF16 Qwen3.8-27B, four MetaX C280 GPUs, one request at a time, temperature 0, thinking explicitly disabled, prefix caching off. The short prompt is 30 tokens on both backends. Numbers below are **completion tokens / whole-request elapsed time**, not steady-state decode TPOT. All comparisons remain subject to the documented incorrect-output failures.

## Size of the gap

| Clean configuration | Output tokens per measured request | Median request time | Output tok/s | vLLM / InfiniLM throughput |
| --- | ---: | ---: | ---: | ---: |
| Original InfiniLM pin | 64 | 9.533 s | 6.713 | 8.13× |
| Register-state prefill kernel, eager policy | 64 | 9.729 s | 6.578 | 8.30× |
| Register kernel + full/piecewise graphs | 64 | 13.069 s | 4.897 | 11.15× |
| Register kernel + decode graphs only | 64 | 8.255 s | 7.753 | 7.04× |
| Fresh vLLM, normal graph-enabled execution | 64 | 1.172 s | 54.601 | 1.00× |

Each row uses three measured requests after warmup where that marker was recorded. The original baseline has three repeats without an explicit warmup marker. These are observations, not confidence intervals. Original InfiniLM takes **8.361 s longer per 64-token request**. The best graph candidate closes only about **15.3% of that elapsed-time gap** and generates incorrect text; it is not an accepted improvement. Corrected eager candidates return 14-token refusals and are excluded from this fixed-length comparison.

The roughly 3.5× prefill improvement does not accelerate the decode kernel: `Qwen3NextGatedDeltaNet::forward` selects `recurrent_gated_delta_rule_indexed` for one-token decode and `chunk_gated_delta_rule` for prefill. Only the latter's implementation was optimized. Its output-throughput result (6.578 versus 6.713 tok/s) shows no decode improvement.

## New confirmed build defect: release label without release optimization

The saved xmake dependency records cover **300 C++ objects**: 211 in `infinicore_runtime`, 87 in `_infinilm`, and two in `_infinicore`. **None has a `-O` optimization flag.** The runtime library still contains out-of-line symbols for tiny `bf16_to_float` and STL-heavy sampling helpers. `InfiniLM/xmake.lua` does not install `mode.release` rules or set optimization explicitly. GCC therefore uses its default optimization level despite the `release` directory name.

The audit's binary hashes match all five measured iteration-06 libraries/modules. This is evidence about the binaries used for the measurements, not just the current source. GPU kernels built separately with `mxcc -O3` are a different build path and already have explicit optimization.

A CPU-only reproduction of the existing BF16 conversion plus greedy argmax over 248,320 logits measured **3.832 ms without an optimization flag versus 0.565 ms with `-O3`** (6.78× faster), with the same selected index. A repeat inside the experiment container (GCC 11.4.0) measured 3.923 ms versus 0.621 ms. This excludes device transfers and does not reproduce the whole inference engine. It demonstrates a real host-build issue; it explains only a few milliseconds per token by itself. **The end-to-end benefit of optimizing the entire C++ runtime has not been measured.** Do not claim that it closes the 8× gap without a controlled rebuild and correctness check.

Evidence: `HOST_BUILD_AUDIT.json`, `sampler_cpu_bench.cpp`, `sampler_default.json`, and `sampler_optimized.json`.

## Where the traced request spends time

The second completed InfiniLM trace window emitted 14 tokens in 2.737 s. It has 1,792 all-reduce kernels on each of four GPUs (128 per forward step). GPU 0's interval union yields the following non-overlapping accounting:

| GPU 0 observation | Time | Fraction of request wall time |
| --- | ---: | ---: |
| No GPU kernel executing | 1,348.58 ms | 49.27% |
| All-reduce kernel residency | 1,087.01 ms | 39.71% |
| Other GPU kernels | 301.60 ms | 11.02% |

This is one GPU's observed timeline, not an additive causal attribution across all ranks. Idle intervals can include dependencies on another rank, host execution, input/output work, and scheduling. They are not automatically removable CPU overhead.

Across aligned collective rounds, median arrival skew is **407.808 us**, while median completion after the final rank arrives is **7.424 us**. **97.88% of summed four-rank collective residency precedes the last arrival**. That strongly supports investigating host launch/rank readiness rather than expecting a different MCCL wire algorithm to remove the gap. The independent 10 KiB BF16 MCCL benchmark also completes correctly in about 14–19 us per rank; the tested FC tuning alternatives were slower.

Grouping 128 reductions per forward identifies a median **112.12 ms from first to last all-reduce within decode**, plus **83.87 ms between successive step boundaries**. The latter includes the LM head, sampling, output handling, next-step preparation, and the next layer's initial work; it is not a pure scheduler timer. The LM-head-shaped GEMV in those gaps takes about **1.55 ms per GPU**. Recorded API durations are generally much smaller than the gaps, but calls can cross interval boundaries and overlap, so API sums are not a latency decomposition.

The trace contains **13 full-vocabulary D2H copies of 496,640 bytes** between steps, matching 248,320 BF16 logits, followed by token-sized transfers. This confirms the host sampling path in `random_sample_metax.cc`. Moving greedy argmax onto the GPU would remove that transfer and host conversion/reduction, but the CPU microbenchmark alone accounts for only a small part of the 83.87 ms gap. The remaining host time needs stage timers or a CPU profile around engine scheduling, tensor/metadata preparation, rank completion, sampling, and detokenization.

Evidence: `OUTPUT_THROUGHPUT.json` and `STEP_GAPS.json`. All these timeline numbers are conditioned on tracing; the tracer increases the standalone MCCL launch-loop overhead materially.

## Kernel and graph divergence

InfiniLM decode replay is confirmed to have **17 device graph segments and 16 host breaks** per rank: 1,075 recorded operators split into 33 segments. MetaX is excluded from `MhaKVCache::supports_device_graph_capture`. `Graph::run` synchronizes before a host segment following device work, so FlashAttention introduces repeated launch/synchronization boundaries. Graph capture has not been made correctness-safe on this model. Simply removing that guard would be unsafe: the linked Torch attention provider creates temporary tensors whose capture/replay lifetime must be handled.

Representative GPU-0 kernel medians in the completed InfiniLM window and the complete **vLLM eager warmup** window are:

| Kernel family | InfiniLM median | vLLM eager warmup median |
| --- | ---: | ---: |
| MCCL FC4 LL BF16 | 409.856 us | 417.024 us |
| Shared mcBLAS GEMV `<256,8,8,4>` | 63.232 us | 62.208 us |
| Shared mcBLAS wave GEMV `<64,2,8>` | 36.352 us | 36.608 us |
| Shared mcBLAS GEMV `<64,8,4,4>` | 16.896 us | 16.640 us |
| Recurrent delta-rule family | 20.480 us | 12.288 us (packed fused decode kernel) |

These are different-length requests with different execution modes, and matching kernel names do not prove matching matrix dimensions. They support prioritization, not a controlled speedup claim. There is no evidence here that those shared GEMV kernels are individually 8× slower. The recurrent family difference suggests a narrower optimization opportunity: at 48 linear-attention layers, the displayed 8.192 us difference corresponds to about **0.39 ms/token**, before considering fusion and launch effects. It cannot explain the whole throughput gap on its own.

Fresh vLLM clean execution is graph-enabled; its new trace deliberately used eager mode and includes JIT warmup. vLLM eager/traced throughput is therefore not its 54.60 tok/s clean reference. Later vLLM trace windows lose GPU events despite complete JSON; long-prefill kernel parity remains unavailable. Full-wrap sums and cross-mode all-reduce ratios must not be used as a wall-time attribution.

## Recommended order of experiments

1. **Correct the C++ release build first**, in a separate candidate: enable xmake's release rules (or explicit optimization), verify `-O2`/`-O3` in recorded compile commands, preserve all device/kernel flags, and compare eager output/logits before measuring performance. This isolates host execution changes from graph changes.
2. Measure real TTFT and generation-step/ITL timings for fixed output lengths; add CPU stage timers to explain the remaining inter-step gap. Keep request-throughput and steady-state decode metrics separate.
3. Stabilize decode graph correctness and the MetaX attention provider's buffer lifetime, then reduce the 16 host breaks. Compare outputs after each graph change.
4. Replace the full-logit host sampling round trip with a numerically checked GPU greedy path; then investigate packed recurrent-delta/gating/normalization fusion. Revisit GEMV dispatch only with matched shapes and complete windows.

No live serving configuration or production binary was changed during this analysis. The analysis establishes an ordering of experiments; it does not certify a converged or production-ready model.
