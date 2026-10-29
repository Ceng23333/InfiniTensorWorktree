#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$ROOT/../.." && pwd)"
SUPPORT="$(cd "$BASE/../.." && pwd)"
INSTALL="$SUPPORT/worktrees/InfiniCore-v20261009-post-refactor-prhead-install"
RESULTS="${RESULTS:-$ROOT/results/padded_replay_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULTS"
mounts=()
if [[ "${USE_INSTALLED:-0}" == 1 ]]; then
  sha256sum "$BASE/InfiniLM/python/infinilm/lib/"*.so "$BASE/InfiniLM/python/infinicore/lib/"*.so "$INSTALL/lib/"*.so > "$RESULTS/libraries.sha256"
else
  cp "$BASE/InfiniLM/build/linux/x86_64/release/"{_infinilm.cpython-312-x86_64-linux-gnu.so,_infinicore.cpython-312-x86_64-linux-gnu.so,libinfinicore_runtime.so} "$RESULTS/"
  mounts+=(-v "$RESULTS/libinfinicore_runtime.so:/opt/infinicore/lib/libinfinicore_runtime.so:ro")
  mounts+=(-v "$RESULTS/_infinilm.cpython-312-x86_64-linux-gnu.so:/base/InfiniLM/python/infinilm/lib/_infinilm.cpython-312-x86_64-linux-gnu.so:ro")
  mounts+=(-v "$RESULTS/_infinicore.cpython-312-x86_64-linux-gnu.so:/base/InfiniLM/python/infinicore/lib/_infinicore.cpython-312-x86_64-linux-gnu.so:ro")
  sha256sum "$RESULTS/"*.so "$INSTALL/lib/"*.so > "$RESULTS/libraries.sha256"
fi
for mode in ${MODES:-eager graph}; do
  docker run --rm --name "$(basename "$RESULTS")-$mode" --runtime dlrt --shm-size=8g \
    -e DENGLIN_DEVICES="${GPU_ID:-1}" -e PYTHONPATH=/base/InfiniLM/python -e TZ=Asia/Shanghai \
    -e LD_LIBRARY_PATH=/usr/local/dlgpu/sdk/lib:/opt/infinicore/lib:/usr/local/lib/python3.12/dist-packages/torch/lib \
    -e ENABLE_GRAPH=1 -e INFINI_PREFILL_FULL_CG=1 -e INFINI_CUDAGRAPH_POLICY=full_and_piecewise \
    -e INFINI_FA_INGRAPH=both -e INFINI_MOE_INGRAPH=both -e INFINI_NATIVE_CG_CAPTURE_BUCKETS=512 \
    -e INFINI_NATIVE_CG_MAX_LAYERS=42 -e INFINI_NATIVE_CG_SKIP_MID_CAPTURE=1 \
    -e INFINI_RANK_WORKER_PROFILE=1 -e INFINI_GRAPH_RUN_PROFILE=1 -e INFINICORE_GRAPH_DEBUG=1 \
    -v "$BASE:/base:ro" -v "$SUPPORT:$SUPPORT:ro" \
    -v /home/qinyiqun/.cache/modelscope:/home/qinyiqun/.cache/modelscope:ro \
    -v /home/qinyiqun/llm_performace_benchmark:/bench:ro -v "$RESULTS:/results:rw" \
    -v "$INSTALL/lib:/opt/infinicore/lib:ro" \
    "${mounts[@]}" \
    --entrypoint /bin/bash dl-vllm-minicpm5-2b-qy:v2026.09.15 -c '
      set -eo pipefail
      source /usr/local/dlgpu/sdk/env.sh
      export CUDA_HOME=/usr/local/dlgpu/sdk
      timeout --kill-after=15s 600s python3 /base/profiling/fp-ab-conc4/padded_replay_probe.py --mode "$1" --output /results 2>&1 | tee "/results/$1.log"
    ' bash "$mode"
done
if [[ "${MODES:-eager graph}" == 'eager graph' ]]; then
  python3 "$ROOT/compare_padded_replay.py" "$RESULTS"
fi
echo "Results: $RESULTS"
