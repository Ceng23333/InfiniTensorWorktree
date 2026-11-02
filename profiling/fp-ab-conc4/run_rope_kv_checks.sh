#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$ROOT/../.." && pwd)"
SUPPORT="$(cd "$BASE/../.." && pwd)"
INSTALL="$SUPPORT/worktrees/InfiniCore-v20261009-post-refactor-prhead-install"
CANDIDATE="${OPS_LIBRARY:-$INSTALL/build/infiniops/src/libinfiniops.so}"
RESULTS="$ROOT/results/rope_kv_checks_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS"
sha256sum "$INSTALL/lib/libinfiniops.so" "$CANDIDATE" > "$RESULTS/provider-sha256.txt"
for variant in installed candidate_off candidate_on; do
  provider="$CANDIDATE"
  [[ "$variant" != installed ]] || provider="$INSTALL/lib/libinfiniops.so"
  flag=0
  [[ "$variant" != candidate_on ]] || flag=1
  docker run --rm --name "$(basename "$RESULTS")-$variant" --runtime dlrt \
    -e DENGLIN_DEVICES="${GPU_ID:-1}" -e TZ=Asia/Shanghai -e PYTHONPATH=/base/InfiniLM/python \
    -e INFINI_CUDAGRAPH_POLICY=full_and_piecewise -e INFINI_GRAPH_RUN_PROFILE=1 \
    -e INFINI_QY_ROPE_SMALL_BLOCK="$flag" -e INFINI_QY_KV_SMALL_BLOCK="$flag" \
    -e LD_LIBRARY_PATH=/usr/local/dlgpu/sdk/lib:/opt/infinicore/lib:/usr/local/lib/python3.12/dist-packages/torch/lib \
    -v "$BASE:/base:ro" -v "$RESULTS:/results:rw" -v "$INSTALL/lib:/opt/infinicore/lib:ro" \
    -v "$provider:/opt/infinicore/lib/libinfiniops.so:ro" \
    -v "$BASE/InfiniLM/build/linux/x86_64/release/_infinicore.cpython-312-x86_64-linux-gnu.so:/base/InfiniLM/python/infinicore/lib/_infinicore.cpython-312-x86_64-linux-gnu.so:ro" \
    --entrypoint /bin/bash dl-vllm-minicpm5-2b-qy:v2026.09.15 -c '
      set -eo pipefail
      source /usr/local/dlgpu/sdk/env.sh
      timeout --kill-after=10s 240s python3 /base/profiling/fp-ab-conc4/kernel_rope_kv_probe.py \
        --check-all --iterations 3 --output "/results/$1.json" 2>&1 | tee "/results/$1.log" | tail -4
    ' bash "$variant"
done
python3 "$ROOT/compare_rope_kv_checks.py" "$RESULTS"
echo "Results: $RESULTS"
