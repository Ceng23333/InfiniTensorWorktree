# QY profiling recovery and fused norm integration

**Correctness update:** matching short serial outputs did not establish model
quality. A later long-context EM gate failed under the original FULL-prefill
policy; see [EM_FINDINGS.md](EM_FINDINGS.md). Model timing results below remain
historical, unqualified smoke measurements.

Date: 2026-10-28, Asia/Shanghai. GPU: KS38 device 1. Image:
`dl-vllm-minicpm5-2b-qy:v2026.09.15`. Existing `libinfiniops.so` is reused unchanged.

## Status

- Profiling now completes normally: both the original separate norm path and
  the fused candidate completed all nine DLPTI counter passes.
- The existing QY `FusedAddRmsNorm` is wired into `RMSNorm::forward_inplace`
  behind **`INFINI_QY_FUSED_ADD_RMSNORM=1`**. Unset or other values retain the
  previous QY behavior. Other backends retain their existing dispatch.
- The existing C++ fused operator binding is registered, with a Python
  `infinicore.ops.add_rms_norm` wrapper for correctness tests and profiling.
- Candidate runtime and binding are built in `InfiniLM/build/linux/x86_64/release`.
  Installed runtime libraries and installed Python extension binaries have **not**
  been replaced. Use the candidate mounts below; the flag alone does not enable
  fusion in an old installed binary. No commit, push, or warehouse mutation.
- Serial smoke output matched on all eight prompts. Concurrency-four outputs
  differed, with repeat variability in both arms. Fusion remains experimental
  pending the broader concurrent model-quality gate.

## Failure resolution scope

The original microbenchmark created the InfiniCore device runtime on the main
thread and left its cleanup until process/TLS shutdown. Existing QY runtime
tests already avoid that lifetime ordering. The microbenchmark now runs on one
owned worker, lets its tensor locals leave scope, synchronizes, switches that
worker to CPU, and joins it while the vendor runtime is still alive. Cleanup
also runs on the exception path; exceptions and profiler failures remain fatal.

This is a **profiling harness lifecycle fix**, not a claim that the underlying
general-purpose QY main-thread TLS teardown fault is repaired. No `os._exit`,
exception suppression, allocator leak, or skipped destructor was introduced.
An ASAN-preloaded minimal main-thread probe exited cleanly, so it did not locate
the original corruption; no allocator root cause is asserted from that probe.

Validation now rejects missing counter records, logged child crashes, and
incomplete `Pass X of Y` sequences, even if DLPTI returns exit code zero.

## Device timing results

Mean summed CU duration per synchronized invocation, five samples per case,
from activity captures (`capture --cache-control none`), not counter replay.
The microbench uses BF16, hidden size 2,048, contiguous inputs. Fused norm resets
both mutable inputs before each measured interval. These are isolated eager
kernel measurements, not full-model or graph wall-clock estimates.

| Tokens | Separate Add + RMSNorm | Reused fused norm | Speedup | vLLM fused norm |
|---|---:|---:|---:|---:|
| 1 | 14.308 us | 9.381 us | 1.53x | 6.561 us |
| 4 | 17.431 us | 9.279 us | 1.88x | 6.755 us |
| 512 | 1854.641 us | 332.723 us | 5.57x | 113.568 us |

The 512-token fused candidate remains about 2.93x slower than vLLM. SiLU was
unchanged at approximately 698 us for 512 tokens, versus vLLM's 271 us.
The next kernel targets remain launch/vectorization behavior, RoPE and attention
selection, rather than treating this fusion as closure of the complete gap.

## Correctness and graphs

`InfiniLM/test/runtime/test_qy_fused_norm.py` passed 36 parameter combinations:
BF16/FP16/FP32, tokens 1/4/512, contiguous/row-padded views, and eager/graph modes.
Each case executes three input variants. Checks cover normalized output against
a float reference and the existing separate operators, exact residual sums,
in-place input/residual aliasing, and untouched padding sentinels. Device graph
replay was observed with `used_device=1`.

BF16 fused versus separate outputs differed at 56 of 12,705,792 checked elements,
within the test tolerance. The existing fused kernel computes `(value * rms) *
weight`, while the separate norm computes `(value * weight) * rms`; this source
difference is consistent with small rounding differences but was not isolated
as the cause of concurrent text differences. The fused kernel stores the rounded
residual before computing variance, preserving the separate path's intermediate
residual rounding. It is not bitwise identical to the separate implementation.

## Full-model smoke runs

Both arms used the existing 42-layer model, BF16, 512-token prompts, 16 output
tokens, FULL-prefill policy, disabled prefix cache and ignored EOS. Source and
model mounts were read-only. `/proc/1/maps` confirms the runtime mount used in
each arm; library hashes are retained. Servers were stopped after testing.

| Workload | Baseline makespan | Fused makespan | Baseline p50 TTFT | Fused p50 TTFT | Exact text matches |
|---|---:|---:|---:|---:|---:|
| Concurrency 4, median of 3 measured batches | 13.455 s | 11.462 s | 1606.914 ms | 1208.704 ms | 0/24 paired requests |
| Concurrency 1, one measured batch | 16.179 s | 15.059 s | 578.852 ms | 443.820 ms | 8/8 requests |

The concurrency-four run is about 14.8% shorter (1.174x throughput at equal
requested tokens); p50 TTFT is about 24.8% lower. These short runs are not the
original 128-output-token benchmark and must not replace its published results.
Six of eight baseline prompts varied across the three concurrency-four repeats.
No quality gain or regression is inferred from these texts alone. Investigate
concurrent scheduling/batching and run the normal quality suite before enabling
fusion by default. The serial sample is only eight short outputs, not a general
quality certification.

## Artifacts

All paths below are relative to this profiling directory:

| Artifact directory under `results/` | Contents |
|---|---|
| `kernel_micro_20261028_095016` | Separate norm counter baseline: 9/9 passes, 126,747 metric rows, valid |
| `kernel_micro_20261028_095621` | Fused norm counters: 9/9 passes, 84,426 metric rows, valid |
| `kernel_micro_20261028_095848` | Clean InfiniLM separate + vLLM activity captures and `device-timings.json` |
| `kernel_micro_20261028_100037` | Clean fused activity capture and `device-timings.json` |
| `fused_model_smoke_20261028_100126` | Concurrency-four runs, texts, maps, hashes, `comparison.json`, preserved first candidate runtime, fused unit-test log |
| `fused_model_smoke_20261028_101026` | Serial runs, texts, maps, hashes, `comparison.json`; uses the rebuilt opt-in runtime |

The first candidate had QY fusion enabled unconditionally; it is preserved only
as the binary used by the first A/B. Current source/build makes it opt-in. Counter
metric names/descriptions remain opaque in this SDK's SQLite export. Counter row
counts establish populated captures, not occupancy/bandwidth conclusions.

## Reproduce

From this profiling directory, with GPU 1 reserved and idle:

```bash
BASE="$(cd ../.. && pwd)"
export RUNTIME_LIBRARY="$BASE/InfiniLM/build/linux/x86_64/release/libinfinicore_runtime.so"
export BINDING_LIBRARY="$BASE/InfiniLM/build/linux/x86_64/release/_infinicore.cpython-312-x86_64-linux-gnu.so"
ENGINES=infinilm NORM_MODE=fused PROFILE_MODE=kprof bash run_kernel_microprofile.sh
ENGINES=infinilm NORM_MODE=fused bash run_kernel_microprofile.sh
bash run_fused_model_smoke.sh
CONCURRENCY=1 RUNS=1 bash run_fused_model_smoke.sh
```

The model smoke runner selects the candidate mount and fusion flag for the fused
arm; direct-op microbenchmarks explicitly call the fused operator and do not use
the module dispatch flag. Standard servers need both the rebuilt runtime and
`INFINI_QY_FUSED_ADD_RMSNORM=1` to opt in.

## Validation limits

- Seven profiling-tool unit tests passed, including interval-based timing,
  optional output text, empty counters, logged failures and partial-pass rejection.
- The 36-case GPU test passed with clean worker shutdown; runtime builds succeeded.
- The broader static runtime-contract suite ran 49 tests with five failures in
  untouched attention/model/graph expectations. Substituting the pre-change
  versions of all three edited tracked runtime/API files in memory reproduced
  the same five failures. They were not rewritten or hidden by this change.
- Full LongBench/EM evaluation, 128-token end-to-end reruns, and a fix for the
  general main-thread QY TLS shutdown fault remain outside the completed checks.
