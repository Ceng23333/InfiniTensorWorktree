#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_WORKTREE="$(cd "${ROOT}/../.." && pwd)"
SUPPORT_ROOT="$(cd "${BASE_WORKTREE}/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-dl-vllm-minicpm5-2b-qy:v2026.09.15}"
MODEL_HOST="${MODEL_HOST:-${SUPPORT_ROOT}/model}"
MODEL_DIR="/models/minicpm5-2b"
CORE_INSTALL="${CORE_INSTALL:-${SUPPORT_ROOT}/worktrees/InfiniCore-v20261009-post-refactor-prhead-install}"
API_PORT="${API_PORT:-18017}"
GPU_ID="${GPU_ID:-1}"
CONTAINER_NAME="${CONTAINER_NAME:-minicpm5-2b-infinilm-full-cg-pti}"
RUN_ID="${RUN_ID:-infinilm_full_cg_pti_$(date +%Y%m%d_%H%M%S)}"
PROFILE_DELAY_MS="${PROFILE_DELAY_MS:-40000}"
RESULTS="${ROOT}/results/${RUN_ID}"

PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
CONCURRENCY="${CONCURRENCY:-4}"
DTYPE="${DTYPE:-bfloat16}"
NUM_BLOCKS="${NUM_BLOCKS:-64}"
MAX_LAYERS="${INFINI_NATIVE_CG_MAX_LAYERS:-42}"

if [[ ! -f "${BASE_WORKTREE}/MANIFEST" || ! -f "${CORE_INSTALL}/lib/libinfinicore_runtime.so" ]]; then
  echo "missing EM-fixed base runtime under ${BASE_WORKTREE} or ${CORE_INSTALL}" >&2
  exit 1
fi
if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

mkdir -p "${RESULTS}"
cp "${BASE_WORKTREE}/MANIFEST" "${RESULTS}/base-MANIFEST.txt"
{
  echo "base_worktree=${BASE_WORKTREE}"
  echo "base_git=$(git -C "${BASE_WORKTREE}" describe --always --dirty)"
  echo "image=${IMAGE_TAG}"
  echo "gpu=${GPU_ID}"
  echo "dtype=${DTYPE}"
  echo "prompt_tokens=${PROMPT_TOKENS}"
  echo "output_tokens=${OUTPUT_TOKENS}"
  echo "concurrency=${CONCURRENCY}"
  echo "prefix_caching=disabled"
  echo "ignore_eos=true"
} > "${RESULTS}/environment.txt"

if [[ ! -f "${RESULTS}/prompts.json" ]]; then
  echo "Building tokenizer-exact prompts..."
  docker run --rm --entrypoint /bin/bash \
    -v "${MODEL_HOST}:${MODEL_DIR}:ro" \
    -v "${BASE_WORKTREE}:/base:rw" \
    "${IMAGE_TAG}" \
    -lc "python3 /base/profiling/fp-ab-conc4/build_prompts.py \
      --model '${MODEL_DIR}' \
      --target-prompt-tokens '${PROMPT_TOKENS}' \
      --output '/base/profiling/fp-ab-conc4/results/${RUN_ID}/prompts.json'"
fi

launch_arm() {
  local mode="$1" graph=0
  local graph_args=()
  if [[ "${mode}" == "full_and_piecewise" ]]; then
    graph=1
    graph_args+=(--enable-graph)
  fi
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
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
    -e PYTHONPATH=/workspace/InfiniLM/python:/workspace/InfiniCore/python \
    -e LD_LIBRARY_PATH=/usr/local/dlgpu/sdk/lib:/opt/infinicore/lib:/usr/local/lib/python3.12/dist-packages/torch/lib \
    -e CUDA_LAUNCH_BLOCKING=0 \
    -e ENABLE_GRAPH="${graph}" \
    -e INFINI_CUDAGRAPH_POLICY=full_and_piecewise \
    -e INFINI_GRAPH_RUN_PROFILE=0 \
    -e INFINI_FA_INGRAPH=both -e INFINI_MOE_INGRAPH=both -e INFINI_PREFILL_FULL_CG=1 \
    -e INFINI_NATIVE_CG_CAPTURE_BUCKETS=512 \
    -e INFINI_NATIVE_CG_MAX_LAYERS="${MAX_LAYERS}" \
    -e INFINI_NATIVE_CG_SKIP_MID_CAPTURE=1 \
    -e INFINI_V1_SCHEDULER=1 \
    -e INFINI_PREFILL_CHUNKED=1 \
    -e INFINI_MAX_NUM_BATCHED_TOKENS=512 \
    -e INFINI_MOE_HOST_TIMER=0 \
    -e INFINI_ROPE_HOST_TIMER=0 \
    -v "${MODEL_HOST}:${MODEL_DIR}:ro" \
    -v "${BASE_WORKTREE}/InfiniLM:/workspace/InfiniLM:ro" \
    -v "${BASE_WORKTREE}/InfiniCore:/workspace/InfiniCore:ro" \
    -v "${CORE_INSTALL}/lib:/opt/infinicore/lib:ro" -v "${BASE_WORKTREE}:/base:rw" \
    "${IMAGE_TAG}" \
    -lc "source /usr/local/dlgpu/sdk/env.sh; export CUDA_HOME=/usr/local/dlgpu/sdk; exec dlpti_tools capture --data-file /base/profiling/fp-ab-conc4/results/${RUN_ID}/infinilm.db --activity-mask cmd,cu,curt --delay ${PROFILE_DELAY_MS} -- python3 /workspace/InfiniLM/python/infinilm/server/inference_server.py \
      --device qy --model=${MODEL_DIR} --tp=1 --host=0.0.0.0 --port=${API_PORT} \
      --dtype=${DTYPE} --temperature=0 --top-p=1 --top-k=1 \
      --max-new-tokens=${OUTPUT_TOKENS} --max-batch-size=${CONCURRENCY} \
      --enable-paged-attn --num-blocks=${NUM_BLOCKS} --block-size=256 \
      --max-cache-len=2048 --attn=flash-attn \
      --ignore-eos --disable-prefix-caching ${graph_args[*]:-}" \
    > "${RESULTS}/${mode}-docker-id.txt"
}

wait_ready() {
  local mode="$1"
  for _ in $(seq 1 180); do
    if curl -sf --connect-timeout 2 --noproxy '*' "http://127.0.0.1:${API_PORT}/health" >/dev/null; then
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
      echo "${mode} container exited" >&2
      docker logs "${CONTAINER_NAME}" 2>&1 | tail -120 >&2
      return 1
    fi
    sleep 5
  done
  echo "${mode} readiness timeout" >&2
  docker logs "${CONTAINER_NAME}" 2>&1 | tail -120 >&2
  return 1
}

run_arm() {
  local mode="$1" tag="$2" tag_id
  tag_id="$(echo "${tag}" | tr '-' '_')"
  echo "=== ${mode} ${tag}: starting server ==="
  launch_arm "${mode}"
  wait_ready "${mode}"
  docker logs -f "${CONTAINER_NAME}" > "${RESULTS}/${mode}-${tag_id}.server.log" 2>&1 &
  local log_pid=$!

  echo "=== ${mode} ${tag}: warming up ==="
  python3 "${ROOT}/bench_client.py" \
    --base-url "http://127.0.0.1:${API_PORT}" \
    --prompts "${RESULTS}/prompts.json" \
    --output "${RESULTS}/${mode}-${tag_id}-warmup.json" \
    --mode "${mode}" \
    --concurrency "${CONCURRENCY}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --warmup-only

  echo "=== ${mode} ${tag}: measuring ==="
  python3 "${ROOT}/bench_client.py" \
    --base-url "http://127.0.0.1:${API_PORT}" \
    --prompts "${RESULTS}/prompts.json" \
    --output "${RESULTS}/${mode}-${tag}.json" \
    --mode "${mode}" \
    --concurrency "${CONCURRENCY}" \
    --output-tokens "${OUTPUT_TOKENS}"

  sleep 2
  docker stop -t 30 "${CONTAINER_NAME}" >/dev/null
  wait "${log_pid}" || true
  docker logs "${CONTAINER_NAME}" > "${RESULTS}/${mode}-${tag_id}.final.log" 2>&1 || true
  sleep 5
}

run_arm full_and_piecewise "pti"

docker run --rm --entrypoint /bin/bash -v "${BASE_WORKTREE}:/base:rw" "${IMAGE_TAG}" \
  -lc "source /usr/local/dlgpu/sdk/env.sh; dlpti_tools export --format json --output /base/profiling/fp-ab-conc4/results/${RUN_ID}/infinilm.json /base/profiling/fp-ab-conc4/results/${RUN_ID}/infinilm.db" | tee "${RESULTS}/export.log"
python3 "${ROOT}/extract_pti_facts.py" --input "${RESULTS}/infinilm.json" --output "${RESULTS}/pti-facts.json"
python3 - "${RESULTS}" <<'PY'
import json, sys
from pathlib import Path
r=Path(sys.argv[1]); x=json.loads((r/'full_and_piecewise-pti.json').read_text())['measured']
facts={'label':'infinilm_full_cg_pti','runs':1,**x}
(r/'target-facts.json').write_text(json.dumps(facts,indent=2)+'\n')
print(json.dumps(facts,indent=2))
PY
echo "Results: ${RESULTS}"
