#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/runtime_env.sh"
MODEL="${MODEL:-/home/metax/Downloads/Qwen3.8-27B}"
TP="${TP:-4}"
NUM_BLOCKS="${NUM_BLOCKS:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
MAX_BATCH="${MAX_BATCH:-1}"
PORT="${PORT:-8000}"
# ENABLE_GRAPH / INFINI_CUDAGRAPH_POLICY come from runtime_env.sh
GRAPH_ARGS=()
if [[ "${ENABLE_GRAPH:-0}" == "1" ]]; then
  GRAPH_ARGS+=(--enable-graph)
fi
cd "$REPO/InfiniLM"
exec /opt/conda/bin/python python/infinilm/server/inference_server.py \
  --device=metax --model="$MODEL" --tp="$TP" \
  --max-new-tokens="${MAX_NEW_TOKENS:-2048}" --max-batch-size="$MAX_BATCH" \
  --num-blocks="$NUM_BLOCKS" --block-size="$BLOCK_SIZE" \
  --enable-paged-attn --attn=flash-attn \
  --disable-prefix-caching \
  "${GRAPH_ARGS[@]}" \
  --host=0.0.0.0 --port="$PORT"
