# Develop this pin on Denglin QY (dl-vllm)

ITW tag: `v2026.10.13-qy-minicpm5-16a3-fa-longctx-cudagraph`

Pins: InfiniCore `82dcc1f…` (`feat/qy-submodules`, Ops `da91f73` longctx kernels), InfiniLM `fdc1a40…` (`feat/qy-backend-build`, cudagraph FA + longctx C1–C3 MoE/RoPE + capture-unsafe RoPE/MulScalar).

Based on cudagraph `v2026.10.10-qy-minicpm5-cudagraph` @ `c0bc441` merged with longctx adapting-lane GPU promotions.

## Image

Reuse unify image `dl-vllm-minicpm5-2b-qy:v2026.09.15`.

## Live-mount loop

DUT: `minicpm5-16a3-qy-inf--post-refactor-longctx` port **18184** / device **3**.

Defaults: `ENABLE_GRAPH=1`, FA+paged, `MAX_CACHE_LEN=8192`,
`INFINICORE_DISABLE_DEVICE_GRAPH_SEGMENTS=1` (documented eager-fallback: MoE/RoPE QY pack paths are not Denglin device-graph capture-safe; FA/gemm recording still enabled via `--enable-graph`).

Install: `InfiniCore-v20261009-post-refactor-longctx-install`.

## Build (QY toolchain)

```bash
export QY_ROOT=/usr/local/denglin
export CUDA_HOME=$QY_ROOT/sdk
cd InfiniLM
python3 scripts/build_infini_stack.py \
  --backend qy \
  --infinicore-root ../InfiniCore \
  --build-root ../../InfiniCore-v20261009-post-refactor-longctx-install/build \
  --allow-dirty-submodules \
  --test
```

## Release gate

Graph-on smoke + EvalScope mixed-4096 P1/N10 EXIT=0 / 100% / no OOR (verified 2026-10-15). Do not require TTFT ≤26 s. Skip LB@7k.
