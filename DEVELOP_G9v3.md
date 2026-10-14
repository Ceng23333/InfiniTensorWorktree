# Develop G9v3-39A5B on MetaX HPCC

ITW branch: `adapt/g9v3-39a5b-metax`

## Pins

| Component | Ref | SHA |
|-----------|-----|-----|
| InfiniCore | `adapt/g9v3-39a5b-metax` (from `v0.2.9`) | `e12f34b69f3403ecdecb889dcf7e5b6c4a26f54d` |
| InfiniLM | `adapt/g9v3-39a5b` (from `origin/main`) | `51ff3a8a0d950fda3fa8edb032663479a9f4caeb` |

Reference for G9 MoE (`minicpm5_moe`): ITW tag `v2026.09.23-qy-minicpm5-16a3` / IL `67c41a74`.

HF `model_type: g9v3` is aliased to `MiniCPM5MoEForCausalLM` (same as `minicpm5_moe`).

## Weights

`/data-aisoft/G9v3-39A5B` → `/nfs/models/G9v3-39A5B` (~73G, 16 shards).

Container note: nested NFS binds under `/models` are not visible inside `infinilm-dev-hpcc37`; use `infinilm-dev-g9v3` which mounts `/nfs/models/G9v3-39A5B` → `/models/G9v3-39A5B`.

## Build (MetaX HPCC)

```bash
source scripts/worktree_env.sh
export INFINI_ROOT=$PWD/.infini   # or /root/.infini-g9v3
# InfiniCore @ adapt/g9v3-39a5b-metax with --metax-gpu=y --aten=y --graph=y --ccl=y
# Stage OpenSSL/Python tarballs under _xmake_pkgs if GitHub is blocked
cd InfiniLM && pip install -e .
```

MetaX Core kernels added for G9 MoE bring-up (on top of v0.2.9):

- `moe_topk_sigmoid` (sigmoid router + top-k, E=320)
- `moe_align`
- `moe_fused_dense` (reference GEMV via `fused_moe` kernels; not CUTLASS)

## Smoke

```bash
# in infinilm-dev-g9v3
export INFINI_ROOT=/workspace/.infini
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/workspace/InfiniLM/python:/workspace/InfiniCore/python
export MACA_VISIBLE_DEVICES=0,1,2,3
python examples/test_infer.py --device metax \
  --model=/models/G9v3-39A5B \
  --tp 4 --max-new-tokens 8 --prompt "你好"
```

Gate (2026-10-14): load 16/16 shards; dense layer-0 + MoE forward; generate completes (`SMOKE_OK`). Prefer TP≥4 on 64GiB Mars X203 when HBM is partially occupied.
