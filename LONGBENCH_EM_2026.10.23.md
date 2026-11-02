# QY MiniCPM5-2B LongBench EM follow-up

Base: release `v2026.10.23-qy-minicpm5-2b-batch-sample-fix`, InfiniLM
`56f9b50d18a78b12b3bda8e1423439eb4ed23b5c`, plus local changes. No new release
or commit has been published. Times below are Asia/Shanghai.

## Confirmed defects

### Batched sampling

`tryGreedyWithInfiniOps` allowed QY `[batch, vocab]` logits and requested
`Argmax(dim=1)`. The pinned InfiniOps QY provider inherits `CudaArgmax`, which
only implements a flattened reduction. Its assertions are disabled in the
installed release build. It wrote one global index and left the remaining
output slots untouched.

A synthetic `[4, 97]` BF16 input with row maxima at `[3, 20, 55, 96]` returned
`[387, -777, -777, -777]` when the output was initialized to `-777`. Calling
the scalar operator per row returned the expected four indices.

The fix performs per-row flattened reductions inside the QY sampling adapter.
The engine can therefore use the batched sampling interface without exposing
unsupported dimensional reduction to the QY provider. NVIDIA behavior is
unchanged. Out-of-vocabulary samples still raise; modulo remapping is not used.

### Chunked prefill KV overwrite

Fixing sampling alone did not recover EM: both four-sample diagnostics still
scored 0/4 and their retained outputs were gibberish. This led to a separate
Python progress bug in `LLMEngine._advance_chunk_progress`.

The processor indexes `req.slot_mapping` relative to
`req.num_local_cached_tokens`. Progress advanced the cached-token count but
kept the original slot list. With 512-token chunks, the second chunk therefore
used slots 0:512 again, overwriting the first chunk's K/V, instead of writing
slots 512:1024. Later chunks repeated this overwrite.

Progress now consumes the corresponding prefix of the slot list when advancing
the cached boundary. It also initializes progress from the cached-prefix length
when `chunk_prefill_offset` is zero, covering budget-limited first chunks with
a reused prefix. The shared helper fixes V1 and legacy chunk progress.

A 32-token inference probe with both fixes produced coherent text for all four
long prompts. The first prompt's output IDs matched exactly between single and
four-request execution. Short chat prompts also matched across single/batched
execution and answered correctly. Evidence:
`result/em-debug-prefix-slots-fixed-20261023.log` under the benchmark root.
All four 32-token prefixes exactly match the retained vLLM wave from
`longbench_task_20261023_144046`; see
`result/em-debug-reference-prefix-comparison-20261023.json`.

## Build provenance

The original deployed `_infinilm` was timestamped October 23 at 10:09, before
the release fix committed at 14:43. A source checkout alone was not a valid
post-fix deployment check. The original binaries were backed up under:

`/home/qinyiqun/llm_performace_benchmark/result/em-debug-binaries-before-20261023`

Rebuilt SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| `libinfinicore_runtime.so` | `b666515c3103567c3f20bd51e345507464af1bc6404b44cca7ce6161ed76d097` |
| `_infinilm.cpython-312-x86_64-linux-gnu.so` | `661995fa692145e2e918b6994351050ad773f5c47630251df96b109e673726f8` |
| `_infinicore.cpython-312-x86_64-linux-gnu.so` | `2287e16f03da20f5977538798255a9f881cd9f3fb90869533f46ab13fa5ccab3` |
| Unchanged `libinfiniops.so` | `09b541a50de2345b7c6ef3627e28d42a29cc620679dc5cf4dc96842ce72dac8b` |

Both diagnostic inference processes were checked through `/proc/PID/maps`:
they loaded the rebuilt runtime from `InfiniLM/build/linux/x86_64/release`,
the rebuilt extension from `InfiniLM/python/infinilm/lib`, and the pinned
provider from `/opt/infinicore/lib/libinfiniops.so`.
The matching runtime is installed in both InfiniLM Python library directories
and `InfiniCore-v20261009-post-refactor-prhead-install/lib`, so the existing
launch scripts do not load the old runtime ahead of the rebuilt modules.

The existing QY image required these build environment settings:

```bash
export QY_ROOT=/usr/local/dlgpu CUDA_HOME=/usr/local/dlgpu/sdk
export INFINI_ROOT=/home/qinyiqun/workspace/minicpm5-2b-support/worktrees/InfiniCore-v20261009-post-refactor-prhead-install/build/prefix
export INFINILM_CXX11_ABI=1 XMAKE_ROOT=y XMAKE_STATS=n
export CPLUS_INCLUDE_PATH=/usr/include/python3.12
export PYBIND11_INCLUDE=/usr/local/lib/python3.12/dist-packages/torch/include
XMAKE=/home/qinyiqun/workspace/minicpm5-2b-support/worktrees/_xmake_bin
"$XMAKE" build -j 4 _infinilm
"$XMAKE" build -j 4 _infinicore
```

## Regression coverage

`InfiniLM/test/runtime/test_random_sample.py` checks scalar and batch sizes
1/2/4, FP16/BF16/FP32, vocabularies 97/130560, repeated calls, and guard values
around a narrowed output view. There are 21 parameter combinations.

The original runtime failed all 12 multi-row combinations. The patched runtime
passes. The fixture owns its device context on a worker and switches back to
CPU before shutdown, avoiding the existing QY TLS teardown fault. Numerical
success alone and exit code 0 alone were insufficient: earlier experiments
could print a teardown crash after reporting successful assertions.

Clean test evidence:
`/home/qinyiqun/llm_performace_benchmark/result/em-debug-sampling-deployed-20261023.log`
(exit 0, no teardown trace, default installed-library search path).

`test/static/test_prefill_chunk_slots.py` executes the actual Python progress
and metadata methods without importing GPU bindings. Three cases failed before
the slot fix; all four pass afterward. They cover successive chunks crossing
noncontiguous physical blocks, an initially cached prefix with a budget-limited
chunk, resumed chunk progress, and an unchanged single-chunk prefill. Together
with the existing chunking tests, all eight chunking tests pass.

The static runtime-contract suite has the same five failure records across
49 tests on HEAD and the modified tree. These cover pre-existing provider
registration, graph lookup, model registration, and graph-safe attention
expectations, not the sampling change.

## LongBench evidence

Results live under `/home/qinyiqun/llm_performace_benchmark/result`.
The scorer, prompts, middle truncation, temperature, output limit (512),
warmup, and concurrency (4) are unchanged.

| Run | Backend | Scope | EM |
| --- | --- | --- | --- |
| `longbench_task_20261023_101313` | Graph | 59 samples, original binary | 9/59 |
| `longbench_task_20261023_122946` | Eager | 59 samples, original binary | 9/59 |
| `longbench_task_20261022_225608` | vLLM | 59 samples, existing reference | 15/59 |
| `longbench_task_20261023_203355` | Eager | First 4, original binary | 0/4 |
| `longbench_task_20261023_205752` | Eager | First 4, sampling fix only | 0/4 |
| `longbench_task_20261023_210002` | Full-and-piecewise | First 4, sampling fix only | 0/4 |
| `longbench_task_20261023_211819` | Eager | First 4, sampling and KV slot fixes | 1/4 |
| `longbench_task_20261023_211610` | Full-and-piecewise | First 4, sampling and KV slot fixes | 1/4 |

Both final runs exited cleanly and predicted `D, D, A, A`, matching the retained
vLLM reference on these four samples (also 1/4). Their input-token hashes match
the reference. Eager and graph outputs are token-for-token identical for all
four requests. They are **not** token-for-token identical to vLLM: common
prefix lengths are 73, 158, 121, and 69 tokens, respectively. Thus the observed
result is recovery of coherent generation and answer-prediction parity on this
subset, not full numerical parity with vLLM.

Graph evidence includes `captured bucket=512 mid=false layers=8 segments=17`
and `graph_run_profile ... used_device=1`. The layer cap of 8 matches the
existing Layer B launcher default; this diagnostic is not a new 42-layer
performance measurement.

Machine-readable comparison, including process exits, EM, input hashes, and
token comparisons:
`/home/qinyiqun/llm_performace_benchmark/result/em-debug-final-comparison-20261023.json`.
The retained vLLM wave is also copied to
`result/em-debug-vllm-reference-wave000-20261023.json` under the benchmark root.

The rebuilt runs retain their raw per-request tokens in `evidence/` and
child-process stdout/stderr in `evidence_logs/`. The normal harness deletes
its intermediate `.work` directory; diagnostic wrapping only preserves
evidence and does not change scoring or generation.

## Full 59-sample verification (October 26)

The complete post-fix rerun used `LIMIT=59`, `CONCURRENCY=4`, chunked prefill
at 512 tokens, `MAX_OUTPUT_TOKENS=512`, warmup enabled, and unchanged scoring.
Both modes completed 15/15 waves with `valid=15`, `invalid=0`, clean exits, and
`status=passed`.

| Run | Backend | EM | Previous EM |
| --- | --- | ---: | ---: |
| `longbench_task_20261026_102033` | Full-and-piecewise | **16/59** | 9/59 (Oct 23); 6/59 (Oct 22) |
| `longbench_task_20261026_123338` | Eager | **17/59** | 9/59 (Oct 23); 4/59 (Oct 22) |
| `longbench_task_20261022_225608` | vLLM reference | 15/59 | unchanged |

Both fixed InfiniLM modes now exceed the stored vLLM reference by one answer.
The absolute gains are +7 graph and +13 eager versus October 23. Full-and-
piecewise logs continue to show the bucket-512 piecewise capture and
`used_device=1` graph replay.

The merged launcher output is
`/home/qinyiqun/llm_performace_benchmark/result/longbench_task_merged_20261026_102006`,
with graph and eager summaries in their respective run directories above.
These results verify the sampling and chunk-slot fixes on the full Layer B set;
they do not establish bit-for-bit parity with vLLM or a new performance record.
