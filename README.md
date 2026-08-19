# InfiniTensorWorktree

Lightweight pin umbrella for **InfiniCore** and **InfiniLM**. Annotated release tags (`vYYYY.MM.DD`) lock the deployment commit (submodule SHAs + `MANIFEST`).

InfiniMetadata was removed from this pin; warehouse identity/metrics live in InfiniOrchestrator (Entrypoint `/metadata`, LoadBalancer `/metrics`). InfiniLM keeps a NoOp fallback when the package is absent.

## Layout

```text
InfiniTensorWorktree/
  InfiniCore/       # submodule
  InfiniLM/         # submodule
  MANIFEST          # ITW_SHA, IC_SHA, IL_SHA
  scripts/          # worktree_env, pin_worktree, release
```

## Clone

```bash
git clone --recurse-submodules https://github.com/Ceng23333/InfiniTensorWorktree.git
cd InfiniTensorWorktree
git checkout vYYYY.MM.DD
git submodule update --init --recursive
source scripts/worktree_env.sh
```

Sibling layout with InfiniOrchestrator (default):

```text
workspace/
  InfiniOrchestrator/
  InfiniTensorWorktree/   # this repo
```

InfiniOrchestrator resolves `INFINI_TENSOR_WORKTREE` to `../InfiniTensorWorktree` unless overridden.

## Release pin

```bash
TAG=vYYYY.MM.DD ./scripts/release.sh --from-current
# or:
IC_SHA=... IL_SHA=... TAG=vYYYY.MM.DD ./scripts/release.sh
git push origin HEAD --tags
```

Manifest fields: `ITW_SHA`, `IC_SHA`, `IL_SHA`, URLs, `PIN_DATE`.

## `--refactor-dev`: Mars device + HPCC toolkit

This directory is the **`refactor-dev` git worktree** (branch `refactor-dev` from tag `v2026.08.18-refactor`). It names the worktree, branch, and container only. It is **not** a new InfiniOrchestrator playground case. Standalone stays `--main` / `--refactor` / `--deploy`.

### Locked analogy

**Mars is to HPCC as NVIDIA is to CUDA.** HPCC is the toolkit (headers, `htcc`, runtime `.so`), not the InfiniRT device name. Do not call this “HPCC driver support.” The backend is **Mars**; it **links HPCC**. Never `--device hpcc`. Never `-DWITH_METAX=ON` on this host. Never alias `MACA_PATH` onto `HPCC_PATH` (that mixes InfiniRT Mars with InfiniOps MetaX).

| Layer | NVIDIA stack | Mars stack (this host) | MetaX stack (classic MACA) |
|-------|----------------|------------------------|----------------------------|
| Device / InfiniRT option | `WITH_NVIDIA`, `kNvidia` | `WITH_MARS`, `kMars` | `WITH_METAX`, `kMetax` |
| Toolkit env | `CUDA_HOME` | `HPCC_PATH` (`/opt/hpcc`) | `MACA_PATH` (`/opt/maca`) |
| Runtime lib | `cudart` | `hcruntime` | `mcruntime` |
| API prefix | `cuda*` | `hc*` (`<hcr/hc_runtime.h>`) | `mc*` (`<mcr/mc_runtime.h>`) |
| InfiniLM flag | `--device nvidia` | `--device mars` | `--device metax` |

Kickoff env: `HPCC_PATH=/opt/hpcc`, **unset** `MACA_*`, cmake `-DAUTO_DETECT_DEVICES=OFF -DWITH_MARS=ON`.

### Container

`infinilm-dev-refactor-dev` (vendor image `1a3cbde5ff2a`, GPU 1, host `8230→8200` / `8231→8201`). Isolated `INFINI_ROOT=/workspace/InfiniLM/build/integration/mars/prefix`. Binds this worktree at `/workspace`, `/nfs:ro` (weights), `/root/zenghua/models:ro` at `/models`. No `/playground` mount.

```bash
./scripts/create-dev-container.sh
source ./scripts/hpcc-env.sh          # inside the container via ./scripts/exec-dev.sh
./scripts/build-mars-stack.sh         # InfiniRT → InfiniOps → InfiniCCL → InfiniLM
./scripts/run-inference-server.sh     # default: --device mars --model /nfs/Qwen3-0.6B
```

First green bar: InfiniRT Mars `ctest` (12/12). First chat bar: `/nfs/Qwen3-0.6B`. Second chat bar: `/models/9g_8b_thinking` (`llama`; `/nfs/9g_8b*` was not present on this host).

Logs: `bench_results/worktree_9g_refactor_dev_20260819T033544Z/`.
