#!/usr/bin/env bash
# Live wall bring-up hints for metax-9 (sequential :8000 swap).
set -euo pipefail
IO="${INFINI_ORCHESTRATOR:-/root/zenghua/workspace/InfiniOrchestrator}"
export MODEL_DIR="${MODEL_DIR:-/data-aisoft/mechdancer/models/Qwen3-8B}"
export API_PORT="${API_PORT:-8000}"
echo "1) Start frozen vLLM: $IO/playground/Standalone/qwen3-8b-mars-vllm--unified-tp4/run-wrap.sh"
echo "2) export BASE_URL_VLLM=http://127.0.0.1:${API_PORT} and measure / keep endpoint"
echo "3) Stop vLLM; start InfiniLM: $IO/playground/Standalone/qwen3-8b-mars-inf--unified-tp4/run-wrap.sh"
echo "   (set INFINI_TENSOR_WORKTREE to deploy pin if desired)"
echo "4) export BASE_URL_INFINILM=http://127.0.0.1:${API_PORT} RUN_WALL=1 SKIP_PROFILER=0 HCTRACER_ROOT=/opt/hpcc"
echo "5) source /root/zenghua/workspace/deploy20261113/autohelix-env.sh && autohelix run -c autohelix/promo-unified-metax9.yaml -n 5"
echo "6) autohelix watch   # other terminal"
echo "7) python scripts/promo/stop_surpass.py  # exit 0 → human ship"
