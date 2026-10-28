# QY RoPE and KV launch promotion

**Correctness update:** the subsequent long-context EM gate exposed an unsafe
FULL-prefill replay configuration. The operator parity checks below remain valid,
but the original model smoke timings are not EM-qualified. See [EM_FINDINGS.md](EM_FINDINGS.md).

Date: 2026-10-28 (Asia/Shanghai). KS38 device 1, SDK 4.2.1, existing
`dl-vllm-minicpm5-2b-qy:v2026.09.15` image. These are opt-in candidates;
installed baseline libraries were not replaced. No commit or push performed.

## Implementation

| Change | Activation | Scope |
|---|---|---|
| Strided RoPE: 1,024 to 64 threads/block | `INFINI_QY_ROPE_SMALL_BLOCK=1` | Existing QY per-item device algorithm, unchanged grid and arithmetic |
| KV reshape/cache: 1,024 to 128 threads/block | `INFINI_QY_KV_SMALL_BLOCK=1` | Existing cache-copy kernel; QY-only launch policy |
| Single-matrix GEMM dispatch restoration | Automatic for QY `batch_count == 1` | Restores installed behavior: `cublasGemmEx`; true batches retain `cublasGemmStridedBatchedEx` |

The launch flags default off, and require the rebuilt provider. Other backends
retain their original KV launch policy. The previous norm candidate remains
separately controlled by `INFINI_QY_FUSED_ADD_RMSNORM=1` in the rebuilt runtime.

Source owners:

- InfiniOps: `src/native/cuda/qy/rope_kernel.cu`, public/native RoPE headers,
  `src/native/cuda/qy/blas.h`, and generic/QY KV launch-policy headers.
- The existing local `src/native/cuda/qy/rope_thread_per_item.cuh` is reused
  unchanged. It is currently untracked and must be included when packaging the
  source changes; it is a required build input.
- InfiniLM: existing paged-caching binding registered and exposed as
  `infinicore.ops.paged_caching` for the probe. No new cache algorithm added.
- This profiling directory: reference/replay probes, bitwise comparison,
  valid-slot timing captures, bounded model clients, and validation drivers.

## Final device timings

Mean summed CU duration per invocation, five samples. BF16, head dimension 128,
16 Q heads and 2 K/V heads, padded/strided stress layouts. RoPE timings include
both Q and K launches. **All KV slots are valid in this final timing run.**
Negative slots and padding sentinels are tested separately by the correctness
matrix. These microbench layouts are not a claim that every model tensor has
identical strides; end-to-end measurements are separate below.

| Operation | Tokens | Installed | Tuned | Speedup |
|---|---:|---:|---:|---:|
| Q + K RoPE | 1 | 8.723 us | 7.222 us | 1.21x |
| Q + K RoPE | 4 | 13.904 us | 7.995 us | 1.74x |
| Q + K RoPE | 512 | 951.211 us | 175.883 us | **5.41x** |
| KV reshape/cache | 1 | 6.067 us | 4.209 us | 1.44x |
| KV reshape/cache | 4 | 6.329 us | 4.396 us | 1.44x |
| KV reshape/cache | 512 | 478.630 us | 64.294 us | **7.44x** |

PTI verifies RoPE blocks `(64,1,1)` and KV blocks `(128,1,1)`. Grids remain
`(tokens,heads,1)` and `(heads,tokens,1)`, respectively. Activity timing is used
for speed comparisons, not counter replay wall time. Both final counter arms
completed **9/9 passes**, each with **126,834 metric rows**. Metric names remain
opaque in the SDK export, so no occupancy or bandwidth claim is inferred from
row counts.

## Correctness and model results

The final matrix passed 54 cases on each of three variants: installed provider,
candidate with flags off, and candidate with flags on. It covers BF16/FP16/FP32,
tokens 1/4/512, contiguous/padded views, GPT-J/GPT-NeoX, random positions,
negative cache slots, page boundaries and untouched padding. Candidate outputs
are bitwise identical to installed outputs for every case. Each case also checks
three graph replays against its eager result; native replay is confirmed in logs.

Full-model smoke: 42 layers, BF16, 512-token prompts, 16 requested output tokens,
FULL prefill, prefix caching disabled, EOS ignored.

| Comparison | Baseline makespan | Candidate makespan | Baseline p50 TTFT | Candidate p50 TTFT | Paired exact texts |
|---|---:|---:|---:|---:|---:|
| RoPE + KV only, concurrency 4, median of 3 batches | 13.422 s | 12.774 s | 1608.410 ms | 1456.727 ms | 18/24 |
| Norm + RoPE + KV, concurrency 1, one batch | 16.295 s | 14.549 s | 576.656 ms | 391.743 ms | 8/8 |

RoPE+KV alone reduced concurrent makespan by about **4.8%** and p50 TTFT by
**9.4%**. The combined serial run was **10.7% shorter**, with **32.1% lower TTFT**.
Concurrent outputs still vary within both arms, so these short tests are not a
full quality gate. The serial result is only eight prompts. These runs do not
replace the original 128-output-token benchmark or LongBench/EM evaluation.

## Rebuild regression found and fixed

The first rebuilt provider made full-model batches approximately 125.8 s instead
of 13.5 s. This was not a RoPE/KV correctness failure. A control isolated GEMM:
single-matrix 512-row execution rose from roughly 0.56 ms to 11.3 ms, including
dispatch/synchronization. The installed binary imports and calls `cublasGemmEx`
behind a batch-count-one branch; the tracked QY BLAS wrapper lacked that branch
and always used the strided-batched API. Symbol inspection and disassembly
confirmed the source/binary mismatch.

Restoring that branch in the QY wrapper restored GEMM timing. Checked cases
with 1, 4 and 512 rows, plus a true batch-count-two case, matched installed
outputs bit-for-bit and passed a float reference check. The model A/B was rerun
successfully after the fix. This restoration preserves baseline performance;
it is **not counted as a new speedup over the installed baseline**.

The rejected binary and two slow completed batches are retained. The third
batch was explicitly stopped; its connection-reset error is expected in that
rejected run. Model clients now have a configurable 180-second process timeout.

## Final artifacts

Paths relative to `results/`:

- `rope_kv_checks_20261028_111824/`: final 54-case comparisons, replay logs,
  provider hashes, preserved `libinfiniops.so` and the updated Python binding.
- `kernel_micro_20261028_111908/`: installed, valid-slot activity capture and
  `device-timings.json`.
- `kernel_micro_20261028_112003/`: tuned, valid-slot activity capture and timings.
- `kernel_micro_20261028_111917/`: installed nine-pass counters.
- `kernel_micro_20261028_112011/`: tuned nine-pass counters.
- `fused_model_smoke_20261028_111200/`: successful RoPE/KV-only concurrency-four A/B.
- `fused_model_smoke_20261028_112058/`: successful combined serial A/B.
- `fused_model_smoke_20261028_104146/`: rejected pre-restoration provider,
  slow batches, and GEMM control data (`gemm-*-checked.json`,
  `gemm-candidate-restored.json`). Do not deploy `rejected-libinfiniops.so`.

Earlier 10:31-10:39 probes used mixed valid/negative KV slots and the pre-GEMM-fix
candidate. They are retained as diagnostics but are not the final timing table.

## Reproduce

From this profiling directory, with GPU 1 reserved/idle:

```bash
bash run_rope_kv_validation.sh
python3 -m unittest discover -s . -p test_kernel_profile.py -v
```

The validation driver expects the candidate already built. It runs the full
correctness matrix, both activity/counter arms, then the combined serial smoke.
For an isolated RoPE/KV-only model comparison:

```bash
BASE="$(cd ../.. && pwd)"
SUPPORT="$(cd "$BASE/../.." && pwd)"
export OPS_LIBRARY="$SUPPORT/worktrees/InfiniCore-v20261009-post-refactor-prhead-install/build/infiniops/src/libinfiniops.so"
export INFINI_QY_ROPE_SMALL_BLOCK=1 INFINI_QY_KV_SMALL_BLOCK=1
FUSED_NORM=0 bash run_fused_model_smoke.sh
```

Provider build uses the existing CMake directory, target `infiniops`, `-j 4`,
inside the pinned image. Mount the workspace at `/host_ws` and the host xmake
cache at `/host_xmake`; set `CMAKE_PREFIX_PATH` to the cached pybind11 3.0.4
package. Build the `_infinicore` xmake target using the environment documented
in `LONGBENCH_EM_2026.10.23.md` for the new Python cache binding. Do not run
`cmake --install` against the baseline prefix as part of candidate testing.

Nine profiling-tool tests pass. GPU correctness, graph replay, GEMM controls,
activity capture, nine-pass counters and short model smoke are complete.
Activation, attention specialization, the larger host/dispatch gap, broad model
quality evaluation and general QY main-thread TLS teardown remain follow-ups.
