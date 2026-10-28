#!/usr/bin/env bash
set -euo pipefail
REPO=/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise
NEW=$REPO/InfiniLM/python/infinilm/lib/libinfinicore_runtime.so
cp -a "$NEW" "$REPO/install/lib/libinfinicore_runtime.so"
cp -a "$NEW" "$REPO/build/integration/metax/prefix/lib/libinfinicore_runtime.so"
# also refresh paired python lib copy used by some loaders
cp -a "$NEW" "$REPO/InfiniLM/python/infinicore/lib/libinfinicore_runtime.so" 2>/dev/null || true
ls -la "$REPO/install/lib/libinfinicore_runtime.so"
strings "$REPO/install/lib/libinfinicore_runtime.so" | grep -F INFINI_FA_NUM_SPLITS
echo DEPLOY_OK
