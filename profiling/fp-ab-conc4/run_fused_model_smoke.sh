#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$ROOT/../.." && pwd)"
SUPPORT="$(cd "$BASE/../.." && pwd)"
IMAGE=dl-vllm-minicpm5-2b-qy:v2026.09.15
RUN_ID="${RUN_ID:-fused_model_smoke_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
RESULTS="$ROOT/results/$RUN_ID"
PORT="${API_PORT:-18029}"
CONCURRENCY="${CONCURRENCY:-4}"
RUNS="${RUNS:-3}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-16}"
CANDIDATE="${RUNTIME_LIBRARY:-$BASE/InfiniLM/build/linux/x86_64/release/libinfinicore_runtime.so}"
mkdir -p "$RESULTS"
exec > >(tee "$RESULTS/runner.log") 2>&1
cp "$ROOT/results/infinilm_full_cg_pti_matched_20261027_1735/prompts.json" "$RESULTS/prompts.json"
sha256sum "$CANDIDATE" "$SUPPORT/worktrees/InfiniCore-v20261009-post-refactor-prhead-install/lib/"*.so > "$RESULTS/library-sha256.txt"
if [[ -n "${OPS_LIBRARY:-}" ]]; then
  sha256sum "$OPS_LIBRARY" >> "$RESULTS/library-sha256.txt"
fi
printf 'candidate_norm=%s\ncandidate_rope=%s\ncandidate_kv=%s\nconcurrency=%s\n' \
  "${FUSED_NORM:-1}" "${INFINI_QY_ROPE_SMALL_BLOCK:-0}" "${INFINI_QY_KV_SMALL_BLOCK:-0}" "$CONCURRENCY" > "$RESULTS/variants.txt"
container=""
cleanup() {
  if [[ -n "$container" ]]; then
    docker stop -t 30 "$container" >/dev/null 2>&1 || true
    docker logs "$container" > "$RESULTS/$arm-server.log" 2>&1 || true
    docker rm "$container" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
for arm in baseline fused; do
  container="$RUN_ID-$arm"
  mounts=()
  fusion=0
  rope=0
  kv=0
  if [[ "$arm" == fused ]]; then
    fusion="${FUSED_NORM:-1}"
    rope="${INFINI_QY_ROPE_SMALL_BLOCK:-0}"
    kv="${INFINI_QY_KV_SMALL_BLOCK:-0}"
    if [[ -n "${OPS_LIBRARY:-}" ]]; then
      mounts+=(-v "$OPS_LIBRARY:/opt/infinicore/lib/libinfiniops.so:ro")
    fi
  fi
  [[ "$arm" != fused ]] || mounts+=(-v "$CANDIDATE:/opt/infinicore/lib/libinfinicore_runtime.so:ro")
  docker run -d --name "$container" --runtime dlrt --entrypoint /bin/bash \
    -p "127.0.0.1:$PORT:$PORT" -e DENGLIN_DEVICES="${GPU_ID:-1}" -e TZ=Asia/Shanghai \
    -e PYTHONPATH=/base/InfiniLM/python \
    -e LD_LIBRARY_PATH=/usr/local/dlgpu/sdk/lib:/opt/infinicore/lib:/usr/local/lib/python3.12/dist-packages/torch/lib \
    -e ENABLE_GRAPH=1 -e INFINI_CUDAGRAPH_POLICY=full_and_piecewise \
    -e INFINI_QY_FUSED_ADD_RMSNORM="$fusion" \
    -e INFINI_QY_ROPE_SMALL_BLOCK="$rope" -e INFINI_QY_KV_SMALL_BLOCK="$kv" \
    -e INFINI_PREFILL_FULL_CG=1 -e INFINI_FA_INGRAPH=both -e INFINI_MOE_INGRAPH=both \
    -e INFINI_NATIVE_CG_CAPTURE_BUCKETS=512 -e INFINI_NATIVE_CG_MAX_LAYERS=42 \
    -e INFINI_NATIVE_CG_SKIP_MID_CAPTURE=1 -e INFINI_V1_SCHEDULER=1 \
    -e INFINI_PREFILL_CHUNKED=1 -e INFINI_MAX_NUM_BATCHED_TOKENS=512 \
    -v "$BASE:/base:ro" -v "$SUPPORT/model:/models/minicpm5-2b:ro" \
    -v "$SUPPORT/worktrees/InfiniCore-v20261009-post-refactor-prhead-install/lib:/opt/infinicore/lib:ro" \
    "${mounts[@]}" "$IMAGE" -c '
      source /usr/local/dlgpu/sdk/env.sh
      export CUDA_HOME=/usr/local/dlgpu/sdk
      exec python3 /base/InfiniLM/python/infinilm/server/inference_server.py \
        --device qy --model=/models/minicpm5-2b --tp=1 --host=0.0.0.0 --port="$1" \
        --dtype=bfloat16 --temperature=0 --top-p=1 --top-k=1 --max-new-tokens="$3" \
        --max-batch-size="$2" --enable-paged-attn --num-blocks=64 --block-size=256 \
        --max-cache-len=2048 --attn=flash-attn --ignore-eos --disable-prefix-caching --enable-graph
    ' bash "$PORT" "$CONCURRENCY" "$OUTPUT_TOKENS"
  ready=0
  for _ in $(seq 1 90); do
    if curl --noproxy '*' -sf "http://127.0.0.1:$PORT/health" >/dev/null; then ready=1; break; fi
    [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]] || break
    sleep 2
  done
  [[ "$ready" == 1 ]] || { echo "$arm failed readiness"; exit 1; }
  docker exec "$container" cat /proc/1/maps > "$RESULTS/$arm-maps.txt"
  for run in $(seq 1 "$RUNS"); do
    timeout --kill-after=10s "${CLIENT_TIMEOUT:-180}s" python3 "$ROOT/bench_client.py" --base-url "http://127.0.0.1:$PORT" \
      --prompts "$RESULTS/prompts.json" --output "$RESULTS/$arm-run$run.json" \
      --mode full_and_piecewise --concurrency "$CONCURRENCY" --output-tokens "$OUTPUT_TOKENS" \
      --max-prompts "${PROMPT_LIMIT:-0}" --ignore-eos --include-output-text
  done
  cleanup
  container=""
done
python3 "$ROOT/compare_fused_model_smoke.py" "$RESULTS"
echo "Results: $RESULTS"
