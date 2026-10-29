#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$ROOT/../.." && pwd)"
SUPPORT="$(cd "$BASE/../.." && pwd)"
INSTALL="$SUPPORT/worktrees/InfiniCore-v20261009-post-refactor-prhead-install"
RUN_ID="${RUN_ID:-em_gate_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
RESULTS="$ROOT/results/$RUN_ID"
mkdir -p "$RESULTS/snapshot"
exec > >(tee "$RESULTS/runner.log") 2>&1
if [[ "${USE_INSTALLED:-0}" == 1 ]]; then
  [[ "${EM_ARMS:-baseline candidate}" == baseline ]] || { echo 'Installed-library run requires EM_ARMS=baseline'; exit 2; }
  cp "$INSTALL/lib/"{libinfinicore_runtime.so,libinfiniops.so} "$RESULTS/snapshot/"
  cp "$BASE/InfiniLM/python/infinilm/lib/_infinilm.cpython-312-x86_64-linux-gnu.so" "$RESULTS/snapshot/"
else
  cp "$BASE/InfiniLM/build/linux/x86_64/release/libinfinicore_runtime.so" "$RESULTS/snapshot/"
  cp "${OPS_LIBRARY:-$INSTALL/build/infiniops/src/libinfiniops.so}" "$RESULTS/snapshot/"
  cp "${ENGINE_LIBRARY:-$BASE/InfiniLM/build/linux/x86_64/release/_infinilm.cpython-312-x86_64-linux-gnu.so}" "$RESULTS/snapshot/"
fi
cp "$BASE/InfiniLM/python/infinicore/lib/_infinicore.cpython-312-x86_64-linux-gnu.so" "$RESULTS/snapshot/"
sha256sum "$RESULTS/snapshot/"*.so "$INSTALL/lib/"*.so > "$RESULTS/library-sha256.txt"
for arm in ${EM_ARMS:-baseline candidate}; do
  [[ "$arm" == baseline || "$arm" == candidate ]] || exit 2
  flag=0
  mounts=()
  if [[ "$arm" == candidate ]]; then
    flag=1
    mounts+=(-v "$RESULTS/snapshot/libinfiniops.so:/opt/infinicore/lib/libinfiniops.so:ro")
  fi
  docker run --rm --name "$RUN_ID-$arm" --runtime dlrt --shm-size=8g \
    -e DENGLIN_DEVICES="${GPU_ID:-1}" -e TZ=Asia/Shanghai -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/base/InfiniLM/python -e CUDA_VISIBLE_DEVICES=0 \
    -e LD_LIBRARY_PATH=/usr/local/dlgpu/sdk/lib:/opt/infinicore/lib:/usr/local/lib/python3.12/dist-packages/torch/lib \
    -e ENABLE_GRAPH=1 -e INFINI_CUDAGRAPH_POLICY=full_and_piecewise -e INFINI_PREFILL_FULL_CG="${PREFILL_FULL_CG:-1}" \
    -e INFINI_FA_INGRAPH=both -e INFINI_MOE_INGRAPH=both -e INFINI_NATIVE_CG_CAPTURE_BUCKETS=512 \
    -e INFINI_NATIVE_CG_MAX_LAYERS=42 -e INFINI_NATIVE_CG_SKIP_MID_CAPTURE=1 \
    -e INFINI_RANK_WORKER_PROFILE="${REPLAY_PROFILE:-0}" -e INFINI_GRAPH_RUN_PROFILE="${REPLAY_PROFILE:-0}" \
    -e INFINI_V1_SCHEDULER=1 -e INFINI_PREFILL_CHUNKED=1 -e INFINI_MAX_NUM_BATCHED_TOKENS=512 \
    -e INFINI_QY_FUSED_ADD_RMSNORM="$flag" -e INFINI_QY_ROPE_SMALL_BLOCK="$flag" -e INFINI_QY_KV_SMALL_BLOCK="$flag" \
    -v "$BASE:/base:ro" -v "$SUPPORT:$SUPPORT:ro" \
    -v /home/qinyiqun/.cache/modelscope:/home/qinyiqun/.cache/modelscope:ro \
    -v /home/qinyiqun/llm_performace_benchmark:/bench:ro -v "$RESULTS:/results:rw" \
    -v "$RESULTS/snapshot/_infinilm.cpython-312-x86_64-linux-gnu.so:/base/InfiniLM/python/infinilm/lib/_infinilm.cpython-312-x86_64-linux-gnu.so:ro" \
    -v "$RESULTS/snapshot/_infinicore.cpython-312-x86_64-linux-gnu.so:/base/InfiniLM/python/infinicore/lib/_infinicore.cpython-312-x86_64-linux-gnu.so:ro" \
    -v "$INSTALL/lib:/opt/infinicore/lib:ro" \
    -v "$RESULTS/snapshot/libinfinicore_runtime.so:/opt/infinicore/lib/libinfinicore_runtime.so:ro" "${mounts[@]}" \
    --entrypoint /bin/bash dl-vllm-minicpm5-2b-qy:v2026.09.15 -c '
      set -eo pipefail
      source /usr/local/dlgpu/sdk/env.sh
      export CUDA_HOME=/usr/local/dlgpu/sdk
      exec python3 /base/profiling/fp-ab-conc4/em_quality_gate.py --benchmark-root /bench --output "/results/$1" --arm "$1" --limit "$2" --diagnostic-output "$3"
    ' bash "$arm" "${LIMIT:-59}" "${EM_DIAGNOSTIC_OUTPUT:-0}"
done
if [[ "${EM_DIAGNOSTIC_OUTPUT:-0}" == 0 && "${LIMIT:-59}" == 59 && "${EM_ARMS:-baseline candidate}" == 'baseline candidate' ]]; then
  python3 "$ROOT/compare_em_gate.py" "$RESULTS"
else
  echo 'No paired-arm comparison: inspect summary.json for single-arm completeness and EM.'
fi
echo "Results: $RESULTS"
