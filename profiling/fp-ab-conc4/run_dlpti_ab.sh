#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_WORKTREE="$(cd "${ROOT}/../.." && pwd)"
SUPPORT_ROOT="$(cd "${BASE_WORKTREE}/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-dl-vllm-minicpm5-2b-qy:v2026.09.15}"
MODEL_HOST="${MODEL_HOST:-${SUPPORT_ROOT}/model}"
MODEL_DIR="/models/minicpm5-2b"
CORE_INSTALL="${CORE_INSTALL:-${SUPPORT_ROOT}/worktrees/InfiniCore-v20261009-post-refactor-prhead-install}"
API_PORT="${API_PORT:-18014}"
GPU_ID="${GPU_ID:-1}"
CONTAINER_NAME="${CONTAINER_NAME:-minicpm5-2b-fp-dlpti}"
RUN_ID="${RUN_ID:-fp_dlpti_$(date +%Y%m%d_%H%M%S)}"
RESULTS="${ROOT}/results/${RUN_ID}"
PROMPTS="${PROMPTS:-${ROOT}/results/prompt-smoke.json}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
CONCURRENCY="${CONCURRENCY:-4}"
DTYPE="${DTYPE:-bfloat16}"
CAPTURE_DELAY_MS="${CAPTURE_DELAY_MS:-80000}"
PARSER="${PARSER:-${ROOT}/fast_dlpti_parse.py}"

if [[ ! -f "${PROMPTS}" || ! -f "${BASE_WORKTREE}/MANIFEST" || ! -x "${PARSER}" ]]; then
  echo "missing prompts, EM-fixed base, or parser" >&2
  exit 1
fi
mkdir -p "${RESULTS}"
cp "${PROMPTS}" "${RESULTS}/prompts.json"
cp "${BASE_WORKTREE}/MANIFEST" "${RESULTS}/base-MANIFEST.txt"
{
  echo "base_worktree=${BASE_WORKTREE}"
  echo "base_git=$(git -C "${BASE_WORKTREE}" describe --always --dirty)"
  echo "image=${IMAGE_TAG}"
  echo "gpu=${GPU_ID}"
  echo "dtype=${DTYPE}"
  echo "capture_delay_ms=${CAPTURE_DELAY_MS}"
  echo "activity_mask=cmd,cu,curt"
  echo "workload=8x512 prompts,128 output tokens,concurrency4"
} > "${RESULTS}/environment.txt"

run_capture() {
  local mode="$1" graph=0
  [[ "${mode}" == "full_and_piecewise" ]] && graph=1
  local graph_args=()
  [[ ${graph} == 1 ]] && graph_args+=(--enable-graph)
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker run -d \
    --name "${CONTAINER_NAME}" --runtime dlrt --hostname "${CONTAINER_NAME}" \
    --entrypoint /bin/bash --workdir /workspace -p "${API_PORT}:${API_PORT}" \
    -e HOME=/home/qinyiqun -e DENGLIN_DEVICES="${GPU_ID}" -e TZ=Asia/Shanghai \
    -e LC_ALL=C.UTF-8 -e LANG=C.UTF-8 \
    -e PATH=/usr/local/dlgpu/sdk/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e PYTHONPATH=/workspace/InfiniLM/python:/workspace/InfiniCore/python \
    -e LD_LIBRARY_PATH=/usr/local/dlgpu/sdk/lib:/opt/infinicore/lib:/usr/local/lib/python3.12/dist-packages/torch/lib \
    -e ENABLE_GRAPH="${graph}" -e INFINI_CUDAGRAPH_POLICY=full_and_piecewise \
    -e INFINI_GRAPH_RUN_PROFILE=0 -e INFINI_FA_INGRAPH=decode \
    -e INFINI_NATIVE_CG_CAPTURE_BUCKETS=512 -e INFINI_NATIVE_CG_MAX_LAYERS=42 \
    -e INFINI_NATIVE_CG_SKIP_MID_CAPTURE=1 -e INFINI_V1_SCHEDULER=1 \
    -e INFINI_PREFILL_CHUNKED=1 -e INFINI_MAX_NUM_BATCHED_TOKENS=512 \
    -e INFINI_MOE_HOST_TIMER=0 -e INFINI_ROPE_HOST_TIMER=0 \
    -v "${MODEL_HOST}:${MODEL_DIR}:ro" \
    -v "${BASE_WORKTREE}/InfiniLM:/workspace/InfiniLM:ro" \
    -v "${BASE_WORKTREE}/InfiniCore:/workspace/InfiniCore:ro" \
    -v "${CORE_INSTALL}/lib:/opt/infinicore/lib:ro" \
    -v "${BASE_WORKTREE}:/base:rw" \
    "${IMAGE_TAG}" \
    -lc "source /usr/local/dlgpu/sdk/env.sh; export CUDA_HOME=/usr/local/dlgpu/sdk; dlpti_tools capture \
      --data-file /base/profiling/fp-ab-conc4/results/${RUN_ID}/${mode}.db \
      --activity-mask cmd,cu,curt --delay ${CAPTURE_DELAY_MS} -- \
      python3 /workspace/InfiniLM/python/infinilm/server/inference_server.py \
      --device qy --model=${MODEL_DIR} --tp=1 --host=0.0.0.0 --port=${API_PORT} \
      --dtype=${DTYPE} --temperature=0 --top-p=1 --top-k=1 \
      --max-new-tokens=${OUTPUT_TOKENS} --max-batch-size=${CONCURRENCY} \
      --enable-paged-attn --num-blocks=64 --block-size=256 --max-cache-len=2048 \
      --attn=flash-attn --ignore-eos --disable-prefix-caching ${graph_args[*]:-}" \
    > "${RESULTS}/${mode}-docker-id.txt"

  for _ in $(seq 1 180); do
    if curl -sf --connect-timeout 2 --noproxy '*' "http://127.0.0.1:${API_PORT}/health" >/dev/null; then
      break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
      docker logs "${CONTAINER_NAME}" 2>&1 | tail -120 > "${RESULTS}/${mode}-startup.log"
      echo "${mode} exited during startup" >&2
      return 1
    fi
    sleep 5
  done

  python3 "${ROOT}/bench_client.py" --base-url "http://127.0.0.1:${API_PORT}" \
    --prompts "${RESULTS}/prompts.json" --output "${RESULTS}/${mode}-warmup.json" \
    --mode "${mode}" --concurrency "${CONCURRENCY}" --output-tokens "${OUTPUT_TOKENS}" --warmup-only
  python3 "${ROOT}/bench_client.py" --base-url "http://127.0.0.1:${API_PORT}" \
    --prompts "${RESULTS}/prompts.json" --output "${RESULTS}/${mode}.json" \
    --mode "${mode}" --concurrency "${CONCURRENCY}" --output-tokens "${OUTPUT_TOKENS}"
  docker logs "${CONTAINER_NAME}" > "${RESULTS}/${mode}.server.log" 2>&1
  docker stop -t 30 "${CONTAINER_NAME}" >/dev/null
  docker rm "${CONTAINER_NAME}" >/dev/null

  docker run --rm --entrypoint /bin/bash -v "${BASE_WORKTREE}:/base:rw" "${IMAGE_TAG}" \
    -lc "source /usr/local/dlgpu/sdk/env.sh; dlpti_tools export --format json \
      --output /base/profiling/fp-ab-conc4/results/${RUN_ID}/${mode}.json \
      /base/profiling/fp-ab-conc4/results/${RUN_ID}/${mode}.db" \
    | tee "${RESULTS}/${mode}-export.log"
}

run_capture eager
run_capture full_and_piecewise

python3 "${PARSER}" summarize "${RESULTS}/eager.json" --json-out "${RESULTS}/eager-summary.json"
python3 "${PARSER}" summarize "${RESULTS}/full_and_piecewise.json" --json-out "${RESULTS}/full_and_piecewise-summary.json"
python3 "${PARSER}" compare "${RESULTS}/eager.json" "${RESULTS}/full_and_piecewise.json" \
  --label-a eager --label-b full_and_piecewise --json-out "${RESULTS}/dlpti-compare.json"
echo "DLPTI results: ${RESULTS}"
