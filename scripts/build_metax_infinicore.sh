#!/usr/bin/env bash
set -euo pipefail
export XMAKE_ROOT=y
export PATH="/root/.local/bin:${PATH}"
export INFINI_ROOT=/root/.infini
export PYTHONUNBUFFERED=1
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export CPLUS_INCLUDE_PATH="${MACA_PATH}/include/mcr:${MACA_PATH}/include/common:${MACA_PATH}/include/mcsparse:${MACA_PATH}/include/mcblas:${MACA_PATH}/include/mcdnn:${MACA_PATH}/include/mcfft:${MACA_PATH}/include/mcrand:${MACA_PATH}/include/mcsolver:${MACA_PATH}/include:${MACA_PATH}/tools/cu-bridge/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="${MACA_PATH}/include/mcr:${MACA_PATH}/include/common:${MACA_PATH}/include/mcsparse:${MACA_PATH}/include/mcblas:${MACA_PATH}/include/mcdnn:${MACA_PATH}/include/mcfft:${MACA_PATH}/include/mcrand:${MACA_PATH}/include/mcsolver:${MACA_PATH}/include:${MACA_PATH}/tools/cu-bridge/include:${C_INCLUDE_PATH:-}"
REPO=/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree
FLASH_SO=/opt/conda/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so
cd "${REPO}/InfiniCore"
echo "START $(date -Is)"
# Reconfigure so include-path patch is picked up; then build/install.
xmake f -y --metax-gpu=y --use-mc=y --aten=y --ccl=y --flash-attn="${FLASH_SO}" -cv
xmake -y
xmake install -y
xmake build -y infiniop-test || true
xmake install -y infiniop-test || true
echo "install.py-equivalent done $(date -Is)"
xmake build -y _infinicore
xmake install -y _infinicore
echo "DONE $(date -Is)"
