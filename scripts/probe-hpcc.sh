#!/usr/bin/env bash
# Probe HPCC toolkit (CUDA analog for Mars). Run inside infinilm-dev-refactor-dev.
set -euo pipefail
HPCC_PATH="${HPCC_PATH:-/opt/hpcc}"
echo "HPCC_PATH=${HPCC_PATH}"
echo "MACA_PATH=${MACA_PATH-<unset>}"
echo "MACA_HOME=${MACA_HOME-<unset>}"
echo "MACA_ROOT=${MACA_ROOT-<unset>}"
echo
if [[ -f "${HPCC_PATH}/Version.txt" ]]; then
  echo "=== Version.txt ==="
  head -5 "${HPCC_PATH}/Version.txt"
  echo
fi
echo "=== runtime libs ==="
ls -l "${HPCC_PATH}/lib"/libhcruntime* 2>/dev/null || echo "no libhcruntime"
ls -l "${HPCC_PATH}/lib"/libmcruntime* 2>/dev/null || echo "no libmcruntime"
ls -l "${HPCC_PATH}/lib"/libhcblas* "${HPCC_PATH}/lib"/libmcblas* 2>/dev/null || true
ls -l "${HPCC_PATH}/lib"/libhcdnn* "${HPCC_PATH}/lib"/libmcdnn* 2>/dev/null || true
echo
echo "=== headers ==="
ls "${HPCC_PATH}/include/hcr/hc_runtime.h" 2>/dev/null || echo "no hcr/hc_runtime.h"
ls "${HPCC_PATH}/include/mcr/mc_runtime.h" 2>/dev/null || echo "no mcr/mc_runtime.h"
echo
echo "=== compilers ==="
ls -d "${HPCC_PATH}/htgpu_llvm" 2>/dev/null || echo "no htgpu_llvm"
ls -d "${HPCC_PATH}/mxgpu_llvm" 2>/dev/null || echo "no mxgpu_llvm"
command -v gcc; gcc --version | head -1
echo
echo "=== tools ==="
command -v ht-smi && ht-smi 2>/dev/null | head -20 || echo "no ht-smi"
command -v mx-smi && mx-smi 2>/dev/null | head -5 || echo "no mx-smi"
echo
echo "=== HPCC_VISIBLE_DEVICES=${HPCC_VISIBLE_DEVICES-} ==="
