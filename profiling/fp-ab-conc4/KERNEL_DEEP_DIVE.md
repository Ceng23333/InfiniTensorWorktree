# Kernel gap investigation and profiling kickoff

Next kernel candidates: [ROPE_KV_PROMOTION.md](ROPE_KV_PROMOTION.md) records
validated RoPE/KV launch tuning and the GEMM fast-path restoration required for
a reproducible provider rebuild.

Follow-up: [FUSED_NORM_FOLLOWUP.md](FUSED_NORM_FOLLOWUP.md) records the resolved
profiling lifecycle failure, complete counter captures, opt-in fused integration,
and full-model smoke comparisons. The kickoff outcomes below are historical.

Date: 2026-10-28 (Asia/Shanghai). KS38 device 1, image
`dl-vllm-minicpm5-2b-qy:v2026.09.15`, existing installed libraries unchanged.

## Findings from measured windows

`kernel_shape_profile.py` streams both original native exports, selects activity
start timestamps within each client's three `measured_wall` intervals, and groups
CU activity by full kernel name, grid, block and CU mask. Statistics outside those
windows remain separate. These are summed activity durations, not critical-path
wall time. The line-oriented parser targets the installed native DLPTI exporter.

The original full-capture tables include initialization/capture and **four**
warmups: the explicit warmup plus one inside each measured client invocation.
Close flash-call counts alone do not establish matched prefill work.

### Shape-specific device durations

Means in microseconds from the original traces, measured windows only. Add + norm
and Q + K RoPE entries sum the respective per-call means; they are not individual
fused-kernel timings. Token-shape interpretations use the model dimensions and
observed grids, not tensor arguments stored in PTI.

| Operation / shape | InfiniLM | vLLM | Ratio |
|---|---:|---:|---:|
| Add + RMSNorm, 4 tokens | 16.04 | 6.01 | 2.67x |
| Add + RMSNorm, 512 tokens | 1907.69 | 113.52 | 16.81x |
| Q + K RoPE, 4 tokens | 12.18 | 6.14 | 1.98x |
| Q + K RoPE, 512 tokens | 949.92 | 108.82 | 8.73x |
| SiLU + multiply, 4 tokens | 8.64 | 6.42 | 1.35x |
| SiLU + multiply, 512 tokens | 696.90 | 270.09 | 2.58x |
| KV reshape/cache, 4 tokens | 5.63 | 3.88 | 1.45x |
| KV reshape/cache, 512 tokens | 479.17 | 53.51 | 8.95x |
| Dominant decode flash launch | 233.80 | 128.14 | 1.82x |

The large Add mean/p50 discrepancy is principally a launch-shape mixture, not
evidence of sporadic 1.6 ms stalls on a 4-token Add. Its 512-token launch itself
averages 1605.41 us; its 4-token launch averages 8.02 us.

### Work amplification and wall-time caveats

- InfiniLM has 1,764 measured 512-token activation launches versus vLLM's 1,008.
  With 42 layers, these are consistent with 42 versus 24 full layer-stack
  executions at that shape: **1.75x work-count amplification**. Scheduler token
  counts and graph bucket padding must distinguish useful work from padding;
  PTI grids alone cannot prove the cause.
- The dominant four-token activation counts are 30,492 versus 31,878. Thus the
  trace is not simply uniformly slower execution of identical launch counts.
- Measured flash proxy totals are 31,388 versus 31,879. This proxy is decode-heavy;
  the backend uses a different varlen attention route for prefill. Do not use it
  as an exact prefill normalization denominator.
- Across all three measured windows, InfiniLM has 20.314 s CU + 17.758 s TU,
  versus vLLM's 6.073 s CU + 15.833 s TU. Client makespans sum to 227.004 s versus
  57.973 s. The 16.165 s difference in summed CU+TU activity does **not** explain
  the 169.032 s wall-time difference. Overlap, uncovered device activity,
  dispatch/synchronization, scheduling and idle intervals need a timeline
  investigation; subtracting these sums is not a CPU-time measurement.
- Replay repetition demonstrates active graph execution, but cannot prove that
  every operation avoids eager fallback or that replay/host overhead is small.

## Source-level targets

1. **QY is excluded from fused norm dispatch.**
   `InfiniLM/csrc/infinicore/src/nn/rmsnorm.cc`, `RMSNorm::forward_inplace`, lists
   fused-capable devices without QY, selecting separate Add and RMSNorm calls.
   A QY `FusedAddRmsNorm` specialization exists under
   `InfiniCore/submodules/InfiniOps/src/native/cuda/qy/ops/fused_add_rms_norm/kernel.h`.
   First test that existing operator and its graph support, rather than assuming
   a new fused implementation is required. BF16 intermediate rounding differs
   between separate and fused normalization; retain model-quality gates.
2. **Generic Add launch/memory behavior.** The QY specialization inherits
   `CudaAdd`; `src/native/cuda/ops/add/kernel.cuh` uses scalar element indexing
   and the launch uses 1,024 threads. The measured 512-token grid is 1,024 blocks.
   Profile contiguous versus production alias/stride layouts, vectorization,
   block size, and metadata/dispatch overhead before selecting an optimization.
3. **RoPE overprovisioning and separate Q/K launches.** The traced QY symbol is
   `ropeThreadPerItemKernel`, with grid `(512,16,1)` / `(512,2,1)` and block 1,024.
   The model's rotary pair count is 64, so the thread-per-pair loop has only 64
   useful lanes for this shape. vLLM uses one Q/K kernel with grid 512 and block
   512. Inspect the installed dispatch: a flat 128-thread RoPE implementation
   also exists in source, but it is not the symbol observed in these traces.
4. **Decode attention selection.** Same dominant grid `(16,4,1)`, different
   blocks: InfiniLM `(4,128,1)`, vLLM `(4,32,1)`. Capture actual strides, cache
   lengths, block tables and selected vendor specialization before attributing
   the difference solely to block size. Prefill uses `mha_varlen_`, decode
   `mha_kvcache_` in `InfiniLM/csrc/layers/attention/backends/flash_attn.cpp`.
5. **Activation and KV cache launch geometry.** At 512 tokens InfiniLM uses
   SiLU grid 3,072 / block 1,024 versus vLLM grid 512 / block 768; KV uses grid
   `(2,512,1)` / block 1,024 versus grid 512 / block 256. The head dimension is
   only 128. Investigate vectorized token-level work and inactive threads.

## Profiling executed

New tools: `kernel_microbench.py`, `run_kernel_microprofile.sh`, and
`validate_kernel_profile.py`. BF16 shapes: tokens 1, 4, 512; hidden 2,048;
intermediate 6,144. Production InfiniLM Add/RMSNorm and SiLU operators are compared
with installed vLLM fused Add/RMSNorm and SiLU. Each case checks numerical output
then performs five bounded invocations, using synchronized intervals. This is an
eager contiguous microbenchmark, not a graph replay or full model performance run.
RMSNorm references reflect each engine's intermediate rounding. Random inputs,
unit norm weights and tolerances 0.03 are smoke checks, not an exhaustive gate.

The runner uses unique disposable containers, GPU 1, read-only source/library
mounts, a 300-second per-arm profiler timeout, and SHA256 library provenance.
No existing serving containers, production kernels or warehouse records changed.

| Run under `results/` | Outcome |
|---|---|
| `kernel_micro_20261028_091251` | Activity exports for both engines; 12/12 operator checks passed. InfiniLM teardown failed, so its export is diagnostic, not a clean capture. |
| `kernel_micro_20261028_091317` | Initial counter filter matched no targets; exports rejected as empty. Retained for provenance. |
| `kernel_micro_20261028_091457` | **Valid vLLM baseline**: 9/9 passes, 84,102 metric rows, 6 launch configurations, 36 invocations, 324 distinct metric-name identifiers. |
| `kernel_micro_20261028_091623` | InfiniLM counter attempt: 26,253 metric rows, pass 1/9 teardown failure; incomplete counters rejected by validation. |

InfiniLM fails with `free(): corrupted unsorted chunks` during
`DevicePinnedHostAllocator` destruction, through runtime/TLS teardown. The stack
is evidence of where the failure surfaces, not proof of the allocator's root
cause. The same symptom occurred in an import/device-initialization probe without
DLPTI; therefore profiler injection alone is not established as its cause.
DLPTI can return exit zero despite `Capture failed: -1`, so the new validator
also checks logs and populated metric tables. The initial crashed import probe
container was explicitly stopped; no background profiling job is left running.

Metric names/descriptions in this SDK's SQLite export are opaque encoded strings.
Use the compatible vendor viewer/decoder to interpret counters; no occupancy,
bandwidth or cache-miss conclusions are claimed from row counts.

## Reproduce and next gates

From this directory, with GPU 1 reserved/idle:

```bash
bash run_kernel_microprofile.sh
PROFILE_MODE=kprof ENGINES=vllm bash run_kernel_microprofile.sh
PROFILE_MODE=kprof ENGINES=infinilm bash run_kernel_microprofile.sh
python3 -m unittest discover -s . -p test_kernel_profile.py
```

The next InfiniLM counter run requires fixing or isolating runtime teardown;
do not suppress the abort or accept pass-1 data as a completed profile. Then
extend isolated probes to RoPE/KV and matched attention arguments, add alias and
stride cases, and compare eager versus graph replay. In parallel with kernel
work, attribute the additional 512-token executions and the unexplained wall-time
gap using scheduler metadata and a CPU/device critical-path timeline.

Measured-window outputs are `kernel-shapes.json` alongside each original
`infinilm.json` / `vllm.json` export. Original trace data is unchanged.

Validation: four focused unit tests passed (window boundaries, shape grouping,
empty counter rejection, child failure rejection and clean populated export),
and `bash -n run_kernel_microprofile.sh` passed. No production rebuild performed.
