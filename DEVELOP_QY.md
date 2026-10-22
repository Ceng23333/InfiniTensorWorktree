# Develop this pin on dl QY (dl-vllm)

ITW tag: `v2026.10.23-qy-minicpm5-2b-full-piecewise` (supersedes `v2026.10.22-…`).

Pins: InfiniCore `82dcc1f…` (`feat/qy-submodules`), InfiniLM `5d5419e…` (`feat/qy-backend-build`: F&P capture-safe MoE/RoPE/MulScalar + **V1 continuous batching + chunked prefill**).

## Image

Reuse unify image `dl-vllm-minicpm5-2b-qy:v2026.09.15`.

## Live-mount loop (2B DUT)

DUT: `minicpm5-2b-qy-inf--post-refactor` port **18003** / `DENGLIN_DEVICES=1`.

Defaults: `ENABLE_GRAPH=1`, `INFINI_CUDAGRAPH_POLICY=full_and_piecewise`, FA+paged,
`INFINI_V1_SCHEDULER=1`, `INFINI_PREFILL_CHUNKED=1`,
`INFINI_MAX_NUM_BATCHED_TOKENS` = `INFINI_NATIVE_CG_CAPTURE_BUCKETS` (default **512**),
`INFINI_NATIVE_CG_SKIP_MID_CAPTURE=1`, `MAX_BATCH_SIZE=1` (raise to 4 for PARALLEL=4 EvalScope).

Kill-switch: `INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS=1`.
Profile: `INFINI_GRAPH_RUN_PROFILE=1`, `INFINICORE_GRAPH_DEBUG=1`.

Install: `InfiniCore-v20261009-post-refactor-prhead-install`.

## CUDA graph policy

| `INFINI_CUDAGRAPH_POLICY` | Behavior |
|---------------------------|----------|
| `eager` | Record ops; replay op-list only (no device segments) |
| `full_and_piecewise` | Segment by `is_device_graph_capture_safe`; capture/replay device segments |
| empty / unknown | Legacy: enable segments unless `INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS` is set |

With CB+chunk, long prompts are multi-step ≤ capture max so prefill hits **PIECEWISE** instead of `over_max` eager.

## Measured P1 (secondary; PARALLEL=1)

| Arm | Avg TTFT | Avg ITL | Staging |
|-----|----------|---------|---------|
| Eager+CB@512 | 4410.8 ms | 89.87 ms | `…_161952` |
| F&P+CB@512 skip-mid | 5031.5 ms | 83.06 ms | `…_163736` |

F&P TTFT may lose vs eager — accepted when EXIT=0 and graphs are live. Primary headline matrix is **PARALLEL=4** EvalScope + Layer B LIMIT=59 / concurrency=4.

## Build (QY toolchain)

```bash
export QY_ROOT=${QY_ROOT:-/usr/local/dlgpu}  # fallback: /usr/local/denglin
export CUDA_HOME=$QY_ROOT/sdk
cd InfiniLM
python3 scripts/build_infini_stack.py \
  --backend qy \
  --infinicore-root ../InfiniCore \
  --build-root ../../InfiniCore-v20261009-post-refactor-prhead-install/build \
  --allow-dirty-submodules \
  --test
```

## Release gate

Primary: EvalScope mixed-4096 **PARALLEL=4** / N10 on 2B — EXIT=0 / 100% / no OOR; decode `used_device≈1` / piecewise capture under F&P. **TTFT may lose vs eager** (policy-active proof).

Also: Layer B LongBench `short_easy_cot` LIMIT=59 concurrency=4 (3 backends); graph-on smoke; prompt > 512 shows multi-step chunk + PIECEWISE.

See InfiniOrchestrator `playground/Standalone/minicpm5-2b-qy-inf--post-refactor/PROFILE_MATRIX.md`.
