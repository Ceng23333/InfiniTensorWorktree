#!/usr/bin/env bash
# Direct InfiniLM inference_server (no InfiniEntrypoint wrap). Default: /nfs/Qwen3-0.6B.
set -euo pipefail

NAME="${CONTAINER_NAME:-infinilm-dev-refactor-dev}"
MODEL="${MODEL:-/nfs/Qwen3-0.6B}"
DEVICE="${DEVICE:-mars}"
PORT="${PORT:-8200}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/exec-dev.sh" "
  mkdir -p '${INFINI_ROOT:-/workspace/InfiniLM/build/integration/mars/prefix}'
  cd /workspace/InfiniLM
  exec python python/infinilm/server/inference_server.py \
    --device '${DEVICE}' \
    --model '${MODEL}' \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port '${PORT}' \
    --tp 1
"
