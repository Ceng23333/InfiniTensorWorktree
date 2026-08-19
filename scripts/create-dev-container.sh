#!/usr/bin/env bash
# Host: create infinilm-dev-refactor-dev bind-mounting this worktree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAME="${CONTAINER_NAME:-infinilm-dev-refactor-dev}"
IMAGE="${BASE_IMAGE:-mx-devops-acr-cn-shanghai.cr.volces.com/pub-registry1/ai-release/hpcc/vllm-mars:0.20.0-hpcc.ai3.7.0.102-torch2.8-py310-kylin2309a-arm64}"
MODELS_DIR="${MODELS_DIR:-/root/zenghua/models}"
NFS_DIR="${NFS_DIR:-/nfs}"
GPU="${HPCC_VISIBLE_DEVICES:-1}"

if ! docker image inspect 1a3cbde5ff2a >/dev/null 2>&1; then
  echo "error: vendor BASE_IMAGE_ID 1a3cbde5ff2a missing" >&2
  exit 1
fi

echo "Removing existing ${NAME} (if any)..."
docker rm -f "${NAME}" >/dev/null 2>&1 || true

echo "Starting ${NAME} from ${IMAGE}..."
docker run -d \
  --name "${NAME}" \
  --privileged \
  --ipc=shareable \
  --shm-size=100g \
  --security-opt apparmor=unconfined \
  --security-opt label=disable \
  --device /dev/dri:/dev/dri \
  --device /dev/htcd:/dev/htcd \
  -p "${HOST_API_PORT:-8230}:8200" \
  -p "${HOST_META_PORT:-8231}:8201" \
  -v "${WORKTREE}:/workspace" \
  -v "${NFS_DIR}:/nfs:ro" \
  -v "${MODELS_DIR}:/models:ro" \
  -e "HPCC_PATH=/opt/hpcc" \
  -e "HPCC_VISIBLE_DEVICES=${GPU}" \
  -e "INFINI_ROOT=/workspace/InfiniLM/build/integration/mars/prefix" \
  -w /workspace \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -lc 'sleep infinity'

echo "CONTAINER_NAME=${NAME}"
echo "WORKTREE=${WORKTREE}"
echo "HPCC_VISIBLE_DEVICES=${GPU}"
echo "HOST_API=http://127.0.0.1:${HOST_API_PORT:-8230}"
docker exec "${NAME}" bash -lc 'uname -m; test -d /opt/hpcc && echo HPCC_OK; test -d /nfs/Qwen3-0.6B && echo QWEN3_0_6B_OK || echo QWEN3_0_6B_MISSING'
