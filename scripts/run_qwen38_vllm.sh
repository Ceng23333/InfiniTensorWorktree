#!/usr/bin/env bash
set -euo pipefail
export MACA_VISIBLE_DEVICES="${MACA_VISIBLE_DEVICES:-2,3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$MACA_VISIBLE_DEVICES}"
MODEL="${MODEL:-/home/metax/Downloads/Qwen3.8-27B}"
TP="${TP:-2}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name Qwen3.8-27B \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --generation-config vllm
