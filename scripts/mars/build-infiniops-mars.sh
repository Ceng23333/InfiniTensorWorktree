#!/usr/bin/env bash
# InfiniOps Mars backend against HPCC toolkit. Do not enable WITH_METAX.
set -euo pipefail

OPS_SRC="${OPS_SRC:-/workspace/InfiniCore/submodules/InfiniOps}"
PREFIX="${INFINI_ROOT:-/workspace/InfiniLM/build/integration/mars/prefix}"
BUILD="${OPS_BUILD:-/workspace/InfiniCore/submodules/InfiniOps/build-mars}"
OPS_JSON="${OPS_JSON:-/workspace/InfiniLM/scripts/configs/infiniops_ops_mars.json}"
JOBS="${JOBS:-$(nproc)}"
HTCC_WRAPPER="${OPS_SRC}/scripts/htcc_wrapper.sh"
PYTHON="${PYTHON:-/opt/conda/bin/python3}"

export HPCC_PATH="${HPCC_PATH:-/opt/hpcc}"
unset MACA_PATH MACA_HOME MACA_ROOT || true
export INFINI_RT_ROOT="${PREFIX}"
# Torch/FA wheels live under conda; keep them on the link search path.
export LD_LIBRARY_PATH="/opt/conda/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${PREFIX}" "${BUILD}"
rm -rf "${BUILD}"
mkdir -p "${BUILD}"

cmake -S "${OPS_SRC}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_C_COMPILER="${HTCC_WRAPPER}" \
  -DCMAKE_CXX_COMPILER="${HTCC_WRAPPER}" \
  -DWITH_CPU=ON \
  -DWITH_MARS=ON \
  -DWITH_METAX=OFF \
  -DWITH_NVIDIA=OFF \
  -DWITH_TORCH=ON \
  -DWITH_LINKED=ON \
  -DAUTO_DETECT_DEVICES=OFF \
  -DAUTO_DETECT_BACKENDS=OFF \
  -DGENERATE_PYTHON_BINDINGS=OFF \
  -DGENERATE_OPERATOR_CALL_INSTANTIATIONS=ON \
  -DPython_EXECUTABLE="${PYTHON}" \
  -DINFINI_RT_ROOT="${PREFIX}" \
  -DINFINI_OPS_OPS="${OPS_JSON}" \
  -DOpenMP_CXX_FLAGS=-fopenmp \
  -DOpenMP_CXX_LIB_NAMES=

cmake --build "${BUILD}" --target infiniops --parallel "${JOBS}"
cmake --install "${BUILD}"
# Ensure runtime finds the rebuilt .so under both lib and lib64.
if [[ -f "${BUILD}/src/libinfiniops.so" ]]; then
  cp -a "${BUILD}/src/libinfiniops.so" "${PREFIX}/lib/libinfiniops.so"
  mkdir -p "${PREFIX}/lib64"
  cp -a "${BUILD}/src/libinfiniops.so" "${PREFIX}/lib64/libinfiniops.so"
fi
echo "Installed InfiniOps Mars (WITH_LINKED FA) → ${PREFIX}"
