#!/usr/bin/env bash
# InfiniOps Mars backend against HPCC toolkit. Do not enable WITH_METAX.
set -euo pipefail

OPS_SRC="${OPS_SRC:-/workspace/InfiniCore/submodules/InfiniOps}"
PREFIX="${INFINI_ROOT:-/workspace/InfiniLM/build/integration/mars/prefix}"
BUILD="${OPS_BUILD:-/workspace/InfiniCore/submodules/InfiniOps/build-mars}"
OPS_JSON="${OPS_JSON:-/workspace/InfiniLM/scripts/configs/infiniops_ops_mars.json}"
JOBS="${JOBS:-$(nproc)}"
HTCC_WRAPPER="${OPS_SRC}/scripts/htcc_wrapper.sh"

export HPCC_PATH="${HPCC_PATH:-/opt/hpcc}"
unset MACA_PATH MACA_HOME MACA_ROOT || true
export INFINI_RT_ROOT="${PREFIX}"

mkdir -p "${PREFIX}" "${BUILD}"

cmake -S "${OPS_SRC}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_C_COMPILER="${HTCC_WRAPPER}" \
  -DCMAKE_CXX_COMPILER="${HTCC_WRAPPER}" \
  -DWITH_CPU=ON \
  -DWITH_MARS=ON \
  -DWITH_METAX=OFF \
  -DWITH_NVIDIA=OFF \
  -DWITH_TORCH=OFF \
  -DWITH_LINKED=OFF \
  -DAUTO_DETECT_DEVICES=OFF \
  -DAUTO_DETECT_BACKENDS=OFF \
  -DGENERATE_PYTHON_BINDINGS=OFF \
  -DINFINI_RT_ROOT="${PREFIX}" \
  -DINFINI_OPS_OPS="${OPS_JSON}"

cmake --build "${BUILD}" --target infiniops --parallel "${JOBS}"
cmake --install "${BUILD}"
echo "Installed InfiniOps Mars → ${PREFIX}"
