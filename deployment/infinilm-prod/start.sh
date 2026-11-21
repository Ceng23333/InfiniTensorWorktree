#!/usr/bin/env bash
# metax-9 (mars) live InfiniLM bring-up for Qwen3-8B fair / AutoHelix promo.
# Editable knobs live here + production-runtime-manifest.json (not Dockerfile alone).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ITW_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- editable serve knobs (env overrides; defaults match fair dual / manifest) ---
MODEL_HOST="${MODEL_DIR:-${MODEL_HOST:-${MODEL:-/data-aisoft/mechdancer/models/Qwen3-8B}}}"
# Accept host path or already-mapped /models/...
if [[ "${MODEL_HOST}" == /models/* ]]; then
  MODEL_CTN="${MODEL_HOST}"
  # Host bind: parent of Qwen3-8B
  MODEL_BIND_SRC="${MODEL_BIND_SRC:-/data-aisoft/mechdancer/models}"
else
  MODEL_BIND_SRC="$(cd "$(dirname "${MODEL_HOST}")" && pwd)"
  MODEL_NAME="$(basename "${MODEL_HOST}")"
  MODEL_CTN="/models/${MODEL_NAME}"
fi

name="${CONTAINER_NAME:-qwen3-8b-mars-inf-tp4}"
image="${IMAGE_TAG:-mx-devops-acr-cn-shanghai.cr.volces.com/pub-registry1/ai-release/hpcc/vllm-mars:0.20.0-hpcc.ai3.7.0.102-torch2.8-py310-kylin2309a-arm64}"
API_PORT="${PORT:-${API_PORT:-8000}}"
CTN_PORT="${CTN_PORT:-8200}"
TP="${TP:-4}"
HPCC_VISIBLE_DEVICES="${HPCC_VISIBLE_DEVICES:-0,1,2,3}"
SHM_SIZE="${SHM_SIZE:-100g}"

MAX_BATCH="${MAX_BATCH:-${MAX_BATCH_SIZE:-64}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
NUM_BLOCKS="${NUM_BLOCKS:-2048}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
ENABLE_GRAPH="${ENABLE_GRAPH:-1}"
INFINI_CUDAGRAPH_POLICY="${INFINI_CUDAGRAPH_POLICY:-legacy}"
INFINI_PREFILL_NATIVE_CG="${INFINI_PREFILL_NATIVE_CG:-0}"
INFINI_FA_INGRAPH="${INFINI_FA_INGRAPH:-decode}"
INFINI_PREFILL_CHUNKED="${INFINI_PREFILL_CHUNKED:-1}"
INFINI_MAX_NUM_BATCHED_TOKENS="${INFINI_MAX_NUM_BATCHED_TOKENS:-2048}"
INFINI_V1_SCHEDULER="${INFINI_V1_SCHEDULER:-1}"
# host_graph_command lever: skip Graph::run tail syncStream (mid-segment syncs remain).
INFINI_GRAPH_SKIP_TAIL_SYNC="${INFINI_GRAPH_SKIP_TAIL_SYNC:-1}"
IGNORE_EOS="${IGNORE_EOS:-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-8B}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1800}"

echo "[infinilm-prod/start] ITW=${ITW_ROOT}"
echo "[infinilm-prod/start] name=${name} image=${image}"
echo "[infinilm-prod/start] model_host=${MODEL_HOST} → ${MODEL_CTN} TP=${TP} PORT=${API_PORT}->${CTN_PORT}"
echo "[infinilm-prod/start] MAX_BATCH=${MAX_BATCH} NUM_BLOCKS=${NUM_BLOCKS} GRAPH=${ENABLE_GRAPH} POLICY=${INFINI_CUDAGRAPH_POLICY}"
echo "[infinilm-prod/start] V1_SCHEDULER=${INFINI_V1_SCHEDULER} SKIP_TAIL_SYNC=${INFINI_GRAPH_SKIP_TAIL_SYNC} MAX_BATCHED_TOKENS=${INFINI_MAX_NUM_BATCHED_TOKENS}"

# Free :API_PORT holders (vLLM fair cell / prior InfiniLM).
if command -v docker >/dev/null 2>&1; then
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker rm -f qwen3-8b-mars-vllm-tp4 >/dev/null 2>&1 || true
  docker rm -f pilotdeck-infinilm-prod >/dev/null 2>&1 || true
fi
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${API_PORT}/tcp" >/dev/null 2>&1 || true
fi
# Long-lived ar-split often holds Qwen3-32B on the same HPCC devices.
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx infinilm-dev-ar-split-mars; then
  echo "[infinilm-prod/start] pausing competing serve in infinilm-dev-ar-split-mars..."
  docker exec infinilm-dev-ar-split-mars bash -lc 'pkill -f "Qwen3-32B" || pkill -f inference_server || true' >/dev/null 2>&1 || true
  sleep 2
fi

docker run -d \
  --name "${name}" \
  --hostname "${name}" \
  --privileged \
  --ipc=shareable \
  --shm-size="${SHM_SIZE}" \
  --security-opt apparmor=unconfined \
  --security-opt label=disable \
  --device /dev/dri:/dev/dri \
  --device /dev/htcd:/dev/htcd \
  -p "${API_PORT}:${CTN_PORT}" \
  -v "${ITW_ROOT}:/workspace" \
  -v /nfs:/nfs:ro \
  -v "${MODEL_BIND_SRC}:/models:ro" \
  -e HPCC_PATH=/opt/hpcc \
  -e "HPCC_VISIBLE_DEVICES=${HPCC_VISIBLE_DEVICES}" \
  -e INFINI_ROOT=/workspace/prefix-mars \
  -e "MODEL=${MODEL_CTN}" \
  -e "TP=${TP}" \
  -e "PORT=${CTN_PORT}" \
  -e "MAX_BATCH=${MAX_BATCH}" \
  -e "MAX_BATCH_SIZE=${MAX_BATCH}" \
  -e "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}" \
  -e "NUM_BLOCKS=${NUM_BLOCKS}" \
  -e "BLOCK_SIZE=${BLOCK_SIZE}" \
  -e "ENABLE_GRAPH=${ENABLE_GRAPH}" \
  -e "INFINI_CUDAGRAPH_POLICY=${INFINI_CUDAGRAPH_POLICY}" \
  -e "INFINI_PREFILL_NATIVE_CG=${INFINI_PREFILL_NATIVE_CG}" \
  -e "INFINI_FA_INGRAPH=${INFINI_FA_INGRAPH}" \
  -e "INFINI_PREFILL_CHUNKED=${INFINI_PREFILL_CHUNKED}" \
  -e "INFINI_MAX_NUM_BATCHED_TOKENS=${INFINI_MAX_NUM_BATCHED_TOKENS}" \
  -e "INFINI_V1_SCHEDULER=${INFINI_V1_SCHEDULER}" \
  -e "INFINI_GRAPH_SKIP_TAIL_SYNC=${INFINI_GRAPH_SKIP_TAIL_SYNC}" \
  -e "IGNORE_EOS=${IGNORE_EOS}" \
  -e "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}" \
  -w /workspace \
  --entrypoint /bin/bash \
  "${image}" \
  -lc 'sleep infinity'

docker exec -d "${name}" bash -lc "
set -eo pipefail
export ENABLE_GRAPH=${ENABLE_GRAPH}
export INFINI_CUDAGRAPH_POLICY=${INFINI_CUDAGRAPH_POLICY}
export INFINI_PREFILL_NATIVE_CG=${INFINI_PREFILL_NATIVE_CG}
export INFINI_FA_INGRAPH=${INFINI_FA_INGRAPH}
export INFINI_PREFILL_CHUNKED=${INFINI_PREFILL_CHUNKED}
export INFINI_MAX_NUM_BATCHED_TOKENS=${INFINI_MAX_NUM_BATCHED_TOKENS}
export INFINI_V1_SCHEDULER=${INFINI_V1_SCHEDULER}
export INFINI_GRAPH_SKIP_TAIL_SYNC=${INFINI_GRAPH_SKIP_TAIL_SYNC}
export PYTHONPATH=/workspace/InfiniLM/python:/workspace/InfiniCore/python:\${PYTHONPATH:-}
source /workspace/scripts/mars/hpcc-env.sh 2>/dev/null || true
export INFINI_ROOT=/workspace/prefix-mars
export LD_LIBRARY_PATH=\"\${INFINI_ROOT}/lib64:\${INFINI_ROOT}/lib:/workspace/InfiniLM/build/linux/arm64/release:\${LD_LIBRARY_PATH:-}\"
export HPCC_VISIBLE_DEVICES=${HPCC_VISIBLE_DEVICES}
cd /workspace/InfiniLM
python3 -m pip install -q janus || pip3 install -q janus || true
GRAPH_ARGS=()
[[ \"${ENABLE_GRAPH}\" == \"1\" ]] && GRAPH_ARGS+=(--enable-graph)
IGNORE_ARGS=()
[[ \"${IGNORE_EOS}\" == \"1\" ]] && IGNORE_ARGS+=(--ignore-eos)
exec python3 python/infinilm/server/inference_server.py \
  --device mars \
  --model ${MODEL_CTN} \
  --host 0.0.0.0 \
  --port ${CTN_PORT} \
  --tp ${TP} \
  --disable-prefix-caching \
  --dtype bfloat16 \
  --enable-paged-attn \
  --max-batch-size ${MAX_BATCH} \
  --max-new-tokens ${MAX_NEW_TOKENS} \
  --num-blocks ${NUM_BLOCKS} \
  --block-size ${BLOCK_SIZE} \
  \"\${GRAPH_ARGS[@]}\" \
  \"\${IGNORE_ARGS[@]}\" \
  > /tmp/qwen3-8b-server.log 2>&1
"

url="http://127.0.0.1:${API_PORT}"
deadline=$((SECONDS + READY_TIMEOUT_SEC))
echo "[infinilm-prod/start] waiting for ${SERVED_MODEL_NAME} at ${url}/v1/models"
while (( SECONDS < deadline )); do
  if curl -sf --noproxy '*' --max-time 5 "${url}/v1/models" 2>/dev/null \
    | grep -q "\"${SERVED_MODEL_NAME}\""; then
    echo "[infinilm-prod/start] ready CONTAINER_NAME=${name} BENCH_TARGET_URL=${url}"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${name}"; then
    echo "error: container exited" >&2
    docker logs "${name}" 2>&1 | tail -80 >&2 || true
    exit 1
  fi
  sleep 5
done

echo "error: timed out waiting for ${SERVED_MODEL_NAME} on :${API_PORT}" >&2
docker exec "${name}" bash -lc 'tail -n 120 /tmp/qwen3-8b-server.log' 2>&1 || true
exit 1
