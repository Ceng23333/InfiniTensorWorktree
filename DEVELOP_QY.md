# Develop this pin on Denglin QY (dl-vllm)

ITW tag: `v2026.10.13-qy-minicpm5-16a3-fa-longctx-cudagraph` (pending Phase 3 green)

Pins: InfiniCore `82dcc1f…` (`feat/qy-submodules`, Ops `da91f73` longctx kernels), InfiniLM `40376ae…` (`feat/qy-backend-build`, cudagraph FA + longctx C1–C3 MoE/RoPE).

Based on cudagraph `v2026.10.10-qy-minicpm5-cudagraph` @ `c0bc441` merged with longctx adapting-lane GPU promotions.

## Image

Reuse unify image `dl-vllm-minicpm5-2b-qy:v2026.09.15`.

## Live-mount loop

Playground DUT: `minicpm5-16a3-qy-inf--post-refactor-longctx` (port **18184** / device **3**), `ENABLE_GRAPH=1`, FA+paged, `MAX_CACHE_LEN=8192`.

Install: `InfiniCore-v20261009-post-refactor-longctx-install` (or sibling longctx-cudagraph-install).

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

Graph-on smoke + EvalScope mixed-4096 P1/N10 EXIT=0 / 100% / no OOR. Do not require TTFT ≤26 s. Skip LB@7k.
