#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export INFINI_ROOT=/root/.infini
export REPO=/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree
export MACA_VISIBLE_DEVICES="${MACA_VISIBLE_DEVICES:-2,3}"
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:/opt/maca/lib:/opt/maca/lib64:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:/opt/conda/lib/python3.10/site-packages:/opt/conda/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export MACA_PATH=/opt/maca

MODEL="${MODEL:-/home/metax/Downloads/Qwen3.8-27B}"
TP="${TP:-2}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
ENABLE_GRAPH="${ENABLE_GRAPH:-1}"

cd "$REPO/InfiniLM"
ARGS=(
  --device=metax
  --model="$MODEL"
  --tp="$TP"
  --max-new-tokens=512
  --max-batch-size=4
  --num-blocks=32
  --block-size=256
  --enable-paged-attn
  --attn=flash-attn
  --disable-prefix-caching
  --host="$HOST"
  --port="$PORT"
)
if [[ "$ENABLE_GRAPH" == "1" ]]; then
  ARGS+=(--enable-graph)
fi
exec /opt/conda/bin/python -m infinilm.server.inference_server "${ARGS[@]}"
