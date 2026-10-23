# Develop this pin on dl QY (dl-vllm)

ITW tag: `v2026.10.23-qy-minicpm5-2b-batch-sample-fix` (supersedes `v2026.10.23-qy-minicpm5-2b-full-piecewise`).

Pins: InfiniCore `82dcc1f…` (`feat/qy-submodules`), InfiniLM `56f9b50…` on branch HEAD (`feat/qy-backend-build`: F&P + V1 CB + chunk; **QY batch_greedy disabled for n_out>1**, `%vocab` soft-map removed).

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
| Graph-off no-CB | 5056.16 ms | 111.37 ms | `…_173052` |

## Measured PARALLEL=4 (primary; MAX_BATCH=4, N10)

| Arm | Avg TTFT | p50 TTFT | Avg ITL | Out toks | EXIT |
|-----|----------|----------|---------|----------|------|
| E4 eager+CB@512 | 14534.59 ms | 9539.3 ms | 308.04 ms | 1027.5 | 10/10 |
| F4 F&P+CB@512 skip-mid | 15800.83 ms | 12287.38 ms | 335.12 ms | 1026.8 | 10/10 |

F4 TTFT may lose vs E4 — accepted when EXIT=0 and graphs are live. See playground `MILESTONE_2026.10.23.md`.

## Layer B (LIMIT=59, concurrency=4) — merged

| Backend | Sample EM | Status |
|---------|-----------|--------|
| infinilm F&P+CB | 6/59 (10.2%) | passed |
| infinilm_eager+CB | 4/59 (6.8%) | passed |
| vllm | 15/59 (25.4%) | passed |

Merged: `longbench_task_merged_20261022_183116/summary.json`.

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
