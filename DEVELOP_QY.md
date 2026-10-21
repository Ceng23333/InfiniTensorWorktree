# Develop this pin on Denglin QY (dl-vllm)

ITW tag: `v2026.10.21-qy-minicpm5-2b-full-piecewise`.

Pins: InfiniCore `82dcc1f…` (`feat/qy-submodules`), InfiniLM `b253606…` (`feat/qy-backend-build`: MoE/RoPE/MulScalar **capture-safe**, `INFINI_CUDAGRAPH_POLICY`, `INFINI_GRAPH_RUN_PROFILE`).

## Image

Reuse unify image `dl-vllm-minicpm5-2b-qy:v2026.09.15`.

## Live-mount loop

DUT: `minicpm5-16a3-qy-inf--post-refactor-longctx` port **18184** / device **3**.

Defaults: `ENABLE_GRAPH=1`, `INFINI_CUDAGRAPH_POLICY=full_and_piecewise`, FA+paged, `MAX_CACHE_LEN=8192`.
Kill-switch: `INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS=1`. Profile: `INFINI_GRAPH_RUN_PROFILE=1`, `INFINICORE_GRAPH_DEBUG=1`.

Install: `InfiniCore-v20261009-post-refactor-prhead-install`.

## CUDA graph policy

| `INFINI_CUDAGRAPH_POLICY` | Behavior |
|---------------------------|----------|
| `eager` | Record ops; replay op-list only (no device segments) |
| `full_and_piecewise` | Segment by `is_device_graph_capture_safe`; capture/replay device segments |
| empty / unknown | Legacy: enable segments unless `INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS` is set |

MoE (`MoeFusedDense` / `MoeAlign*` / `MoeTopkSigmoid`), `RoPE`, and `MulScalar` are capture-safe (vLLM / pre-refactor proven); impls skip `cudaStreamSynchronize` while capturing.

## Build (QY toolchain)

```bash
export QY_ROOT=/usr/local/denglin
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

Graph-on smoke + EvalScope mixed-4096 P1/N10 EXIT=0 / 100% / no OOR.
Expect decode `segmented graph: … host_segments≈0` / `segments≈1` with `INFINICORE_GRAPH_DEBUG=1`.
Do not require TTFT ≤26 s. Skip LB@7k.
See InfiniOrchestrator `playground/Standalone/minicpm5-16a3-qy-inf--post-refactor-longctx/PROFILE_MATRIX.md`.
