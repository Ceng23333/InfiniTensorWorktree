#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$ROOT/../.." && pwd)"
SUPPORT="$(cd "$BASE/../.." && pwd)"
PROVIDER="$SUPPORT/worktrees/InfiniCore-v20261009-post-refactor-prhead-install/build/infiniops/src/libinfiniops.so"
export BINDING_LIBRARY="$BASE/InfiniLM/build/linux/x86_64/release/_infinicore.cpython-312-x86_64-linux-gnu.so"
bash "$ROOT/run_rope_kv_checks.sh"
export ENGINES=infinilm MICRO_SCRIPT=kernel_rope_kv_probe.py
for variant in installed tuned; do
  unset OPS_LIBRARY
  export INFINI_QY_ROPE_SMALL_BLOCK=0 INFINI_QY_KV_SMALL_BLOCK=0
  if [[ "$variant" == tuned ]]; then
    export OPS_LIBRARY="$PROVIDER" INFINI_QY_ROPE_SMALL_BLOCK=1 INFINI_QY_KV_SMALL_BLOCK=1
  fi
  for mode in capture kprof; do
    PROFILE_MODE="$mode" bash "$ROOT/run_kernel_microprofile.sh"
  done
done
FUSED_NORM=1 CONCURRENCY=1 RUNS=1 bash "$ROOT/run_fused_model_smoke.sh"
