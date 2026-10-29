#!/usr/bin/env bash
set -euo pipefail
export MACA_VISIBLE_DEVICES=0,1,2,3
extra=()
if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then extra+=(--enforce-eager); fi
exec python -m vllm.entrypoints.openai.api_server \
 --model /home/metax/Downloads/Qwen3.8-27B --served-model-name Qwen3.8-27B \
 --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 32768 \
 --gpu-memory-utilization 0.85 --trust-remote-code --host 0.0.0.0 --port 8000 \
 --no-enable-prefix-caching --generation-config vllm "${extra[@]}"
