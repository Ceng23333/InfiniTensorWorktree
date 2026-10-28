#!/usr/bin/env bash
set -euo pipefail
REPO=/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise
OBJ=$REPO/InfiniLM/build/.objs/infinicore_runtime/linux/x86_64/release/csrc/infinicore/src/ops/mha_kvcache/mha_kvcache_infiniops.cc.o
rm -f "$OBJ"
bash "$REPO/scripts/rebuild_infinilm_only.sh"
echo REBUILD_DONE
