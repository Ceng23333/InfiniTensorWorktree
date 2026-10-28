#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_WORKTREE="$(cd "${ROOT}/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-dl-vllm-minicpm5-2b-qy:v2026.09.15}"
MODEL_HOST="${MODEL_HOST:-$(readlink -f "${BASE_WORKTREE}/../../model")}"
API_PORT="${API_PORT:-18015}"
GPU_ID="${GPU_ID:-1}"
CONTAINER_NAME="${CONTAINER_NAME:-minicpm5-2b-vllm-pti-target}"
RUN_ID="${RUN_ID:-vllm_pti_target_$(date +%Y%m%d_%H%M%S)}"
RESULTS="${ROOT}/results/${RUN_ID}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
PROFILE="${PROFILE:-1}"
DELAY_MS="${DELAY_MS:-10000}"
CONCURRENCY="${CONCURRENCY:-4}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"

if [[ ! -f "${ROOT}/results/prompt-smoke.json" ]]; then
  echo "missing tokenizer-exact prompts: ${ROOT}/results/prompt-smoke.json" >&2
  exit 1
fi
mkdir -p "${RESULTS}"
cp "${ROOT}/results/prompt-smoke.json" "${RESULTS}/prompts.json"
if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

{
  echo "purpose=vllm_pti_target"
  echo "image=${IMAGE_TAG}"
  echo "sdk=4.2.1"
  echo "vllm=0.21.0"
  echo "gpu=${GPU_ID}"
  echo "model=$(basename "$(readlink -f "${MODEL_HOST}")")"
  echo "requests=8"
  echo "prompt_tokens=512"
  echo "output_tokens=${OUTPUT_TOKENS}"
  echo "concurrency=${CONCURRENCY}"
  echo "max_num_seqs=4"
  echo "max_num_batched_tokens=512"
  echo "prefix_caching=disabled"
  echo "ignore_eos=true"
  echo "enforce_eager=${ENFORCE_EAGER}"
  echo "dlpti_profile=${PROFILE}"
  echo "dlpti_delay_ms=${DELAY_MS}"
} > "${RESULTS}/environment.txt"

EAGER_ARGS=()
[[ "${ENFORCE_EAGER}" == "1" ]] && EAGER_ARGS+=(--enforce-eager)
PTI_ARGS=()
if [[ "${PROFILE}" == "1" ]]; then
  PTI_ARGS+=(
    dlpti_tools capture
    --data-file "/base/profiling/fp-ab-conc4/results/${RUN_ID}/vllm.db"
    --activity-mask cmd,cu,curt
    --delay "${DELAY_MS}"
    --
  )
fi

docker run -d \
  --name "${CONTAINER_NAME}" \
  --runtime dlrt \
  --hostname "${CONTAINER_NAME}" \
  --entrypoint /bin/bash \
  --workdir /workspace \
  -p "${API_PORT}:${API_PORT}" \
  -e HOME=/home/qinyiqun \
  -e DENGLIN_DEVICES="${GPU_ID}" \
  -e TZ=Asia/Shanghai \
  -e LC_ALL=C.UTF-8 \
  -e LANG=C.UTF-8 \
  -e PATH=/usr/local/dlgpu/sdk/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -e LD_LIBRARY_PATH=/usr/local/dlgpu/sdk/lib:/usr/local/lib/python3.12/dist-packages/torch/lib \
  -e PYTHONPATH= \
  -e VLLM_USE_V1=1 \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_LOG_STATS_LEVEL=INFO \
  -v "${MODEL_HOST}:/model:ro" \
  -v "${BASE_WORKTREE}:/base:rw" \
  "${IMAGE_TAG}" \
  -lc "source /usr/local/dlgpu/sdk/env.sh; export CUDA_HOME=/usr/local/dlgpu/sdk; exec ${PTI_ARGS[*]:-} python3 -m vllm.entrypoints.openai.api_server \
    --model /model --served-model-name minicpm5-2b --chat-template /model/chat_template.jinja \
    --host 0.0.0.0 --port ${API_PORT} --generation-config vllm \
    --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 512 \
    --enable-chunked-prefill --no-enable-prefix-caching ${EAGER_ARGS[*]:-}" \
  > "${RESULTS}/docker-id.txt"

ready=0
for _ in $(seq 1 240); do
  if curl -sf --connect-timeout 2 --noproxy '*' "http://127.0.0.1:${API_PORT}/v1/models" >/dev/null; then
    ready=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "vLLM container exited during startup" >&2
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -200 >&2
    exit 1
  fi
  sleep 5
done
if [[ ${ready} != 1 ]]; then
  echo "vLLM readiness timeout" >&2
  docker logs "${CONTAINER_NAME}" 2>&1 | tail -200 >&2
  exit 1
fi

docker logs -f "${CONTAINER_NAME}" > "${RESULTS}/vllm.server.log" 2>&1 &
log_pid=$!
curl -sS --noproxy '*' "http://127.0.0.1:${API_PORT}/version" > "${RESULTS}/version.json"
curl -sS --noproxy '*' "http://127.0.0.1:${API_PORT}/v1/models" > "${RESULTS}/models.json"
curl -sS --noproxy '*' "http://127.0.0.1:${API_PORT}/server_info" > "${RESULTS}/server-info.json" || true
curl -sS --noproxy '*' "http://127.0.0.1:${API_PORT}/metrics" > "${RESULTS}/metrics.before-warmup.txt"

python3 "${ROOT}/bench_client.py" \
  --base-url "http://127.0.0.1:${API_PORT}" \
  --prompts "${RESULTS}/prompts.json" \
  --output "${RESULTS}/warmup.json" \
  --mode eager \
  --concurrency "${CONCURRENCY}" \
  --output-tokens "${OUTPUT_TOKENS}" \
  --ignore-eos --warmup-only
curl -sS --noproxy '*' "http://127.0.0.1:${API_PORT}/metrics" > "${RESULTS}/metrics.after-warmup.txt"

for run in 1 2 3; do
  python3 "${ROOT}/bench_client.py" \
    --base-url "http://127.0.0.1:${API_PORT}" \
    --prompts "${RESULTS}/prompts.json" \
    --output "${RESULTS}/vllm-run${run}.json" \
    --mode vllm_graph \
    --concurrency "${CONCURRENCY}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --ignore-eos
  curl -sS --noproxy '*' "http://127.0.0.1:${API_PORT}/metrics" > "${RESULTS}/metrics.after-run${run}.txt"
done

sleep 2
docker stop -t 30 "${CONTAINER_NAME}" >/dev/null
wait "${log_pid}" || true
docker logs "${CONTAINER_NAME}" > "${RESULTS}/vllm.final.log" 2>&1 || true
docker rm "${CONTAINER_NAME}" >/dev/null

if [[ "${PROFILE}" == "1" ]]; then
  docker run --rm --entrypoint /bin/bash -v "${BASE_WORKTREE}:/base:rw" "${IMAGE_TAG}" \
    -lc "source /usr/local/dlgpu/sdk/env.sh; dlpti_tools export --format json \
      --output /base/profiling/fp-ab-conc4/results/${RUN_ID}/vllm.json \
      /base/profiling/fp-ab-conc4/results/${RUN_ID}/vllm.db" \
    | tee "${RESULTS}/export.log"
  python3 "${ROOT}/extract_pti_facts.py" \
    --input "${RESULTS}/vllm.json" \
    --output "${RESULTS}/pti-facts.json"
fi

python3 "${ROOT}/summarize_target.py" --results "${RESULTS}" --label vllm_graph
echo "Results: ${RESULTS}"
