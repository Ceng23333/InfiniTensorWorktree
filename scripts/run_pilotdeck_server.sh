#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export INFINI_ROOT=/root/.infini
export REPO=/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree
# tp=4 on all 4 GPUs leaves enough VRAM for PilotDeck agent+tool KV (~8k+ tokens).
export MACA_VISIBLE_DEVICES="${MACA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$REPO/InfiniCore/python/infinicore/lib:$REPO/InfiniLM/python/infinilm/lib:/opt/maca/lib:/opt/maca/lib64:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:/opt/conda/lib/python3.10/site-packages:/opt/conda/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export MACA_PATH=/opt/maca

MODEL="${MODEL:-/home/metax/Downloads/Qwen3.8-27B}"
TP="${TP:-4}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
ENABLE_GRAPH="${ENABLE_GRAPH:-1}"
# Default 32 (~8k tokens) is too small for PilotDeck agent+tool schemas (~9.6k est.).
# Override with NUM_BLOCKS / BLOCK_SIZE. 64×256 ≈ 16k tokens fits first-turn GUI chat.
NUM_BLOCKS="${NUM_BLOCKS:-64}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"

cd "$REPO/InfiniLM"
ARGS=(
  --device=metax
  --model="$MODEL"
  --tp="$TP"
  --max-new-tokens=512
  --max-batch-size=4
  --num-blocks="$NUM_BLOCKS"
  --block-size="$BLOCK_SIZE"
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
