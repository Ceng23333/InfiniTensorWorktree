#!/usr/bin/env bash
# Direct InfiniLM inference_server on Mars (no MetaX/MACA). Default: Qwen3-32B TP4.
set -euo pipefail

NAME="${CONTAINER_NAME:-infinilm-dev-ar-split-mars}"
MODEL="${MODEL:-/models/Qwen3-32B}"
DEVICE="${DEVICE:-mars}"
PORT="${PORT:-8200}"
TP="${TP:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONTAINER_NAME="${NAME}"
export HPCC_VISIBLE_DEVICES="${HPCC_VISIBLE_DEVICES:-0,1,2,3}"
export INFINI_ROOT="${INFINI_ROOT:-/workspace/prefix-mars}"

# Prod-like knobs: paged FA + graph (prefix cache off).
EXTRA_ARGS="${EXTRA_ARGS:---disable-prefix-caching --dtype bfloat16 --enable-paged-attn}"
if [[ "${ENABLE_GRAPH:-1}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --enable-graph"
fi

exec "${SCRIPT_DIR}/exec-dev.sh" "
  export INFINI_PREFILL_CHUNKED=\${INFINI_PREFILL_CHUNKED:-1}
  export INFINI_MAX_NUM_BATCHED_TOKENS=\${INFINI_MAX_NUM_BATCHED_TOKENS:-2048}
  export ENABLE_GRAPH=\${ENABLE_GRAPH:-1}
  export INFINI_CUDAGRAPH_POLICY=\${INFINI_CUDAGRAPH_POLICY:-legacy}
  export INFINI_FA_INGRAPH=\${INFINI_FA_INGRAPH:-decode}
  mkdir -p '\${INFINI_ROOT}'
  cd /workspace/InfiniLM
  exec python3 python/infinilm/server/inference_server.py \
    --device '${DEVICE}' \
    --model '${MODEL}' \
    --host 0.0.0.0 \
    --port '${PORT}' \
    --tp '${TP}' \
    ${EXTRA_ARGS}
"
