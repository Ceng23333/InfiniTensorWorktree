#!/usr/bin/env bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
export PYTHONUNBUFFERED=1
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export MACA_VISIBLE_DEVICES="${MACA_VISIBLE_DEVICES:-0,1,2,3}"
export INFINI_ROOT="${INFINI_ROOT:-$REPO/install}"
export PATH="/root/.local/bin:/opt/conda/bin:$MACA_PATH/bin:${PATH:-}"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$INFINI_ROOT/lib64:/opt/conda/lib/python3.10/site-packages/torch/lib:$MACA_PATH/lib:$MACA_PATH/lib64:$MACA_PATH/ompi/lib:$MACA_PATH/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/InfiniLM/python:${PYTHONPATH:-}"
# F&P defaults (product)
export ENABLE_GRAPH=${ENABLE_GRAPH:-1}
export INFINI_CUDAGRAPH_POLICY=${INFINI_CUDAGRAPH_POLICY:-eager}
export INFINI_V1_SCHEDULER="${INFINI_V1_SCHEDULER:-1}"
export INFINI_PREFILL_CHUNKED="${INFINI_PREFILL_CHUNKED:-1}"
export INFINI_NATIVE_CG_SKIP_MID_CAPTURE="${INFINI_NATIVE_CG_SKIP_MID_CAPTURE:-1}"
export INFINI_NATIVE_CG_CAPTURE_BUCKETS="${INFINI_NATIVE_CG_CAPTURE_BUCKETS:-2048}"
export INFINI_MAX_NUM_BATCHED_TOKENS="${INFINI_MAX_NUM_BATCHED_TOKENS:-$INFINI_NATIVE_CG_CAPTURE_BUCKETS}"
# OpenMP: InfiniOps needs LLVM __kmpc_* (torch ships libomp)
_TORCH_LIB="/opt/conda/lib/python3.10/site-packages/torch/lib"
if [ -d "$_TORCH_LIB" ]; then
  export LD_LIBRARY_PATH="$_TORCH_LIB:$LD_LIBRARY_PATH"
fi
for _omp in "$_TORCH_LIB/libomp.so" "$_TORCH_LIB/libiomp5.so" /opt/conda/lib/libomp.so /opt/conda/lib/libiomp5.so; do
  if [ -f "$_omp" ]; then
    export LD_PRELOAD="${_omp}${LD_PRELOAD:+:$LD_PRELOAD}"
    break
  fi
done
# Stub LLVM OpenMP symbols expected by MetaX-built InfiniOps
if [ -f "$INFINI_ROOT/lib/libkmpc_stub.so" ]; then
  export LD_PRELOAD="$INFINI_ROOT/lib/libkmpc_stub.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export INFINI_FA_NUM_SPLITS=${INFINI_FA_NUM_SPLITS:-1}
export INFINI_ENABLE_THINKING=${INFINI_ENABLE_THINKING:-0}
