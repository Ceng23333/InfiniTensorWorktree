#!/usr/bin/env bash
# InfiniCCL Mars device + HCCL against HPCC toolkit. Do not enable WITH_METAX.
set -euo pipefail

CCL_SRC="${CCL_SRC:-/workspace/InfiniCore/submodules/InfiniCCL}"
PREFIX="${INFINI_ROOT:-/workspace/InfiniLM/build/integration/mars/prefix}"
BUILD="${CCL_BUILD:-/workspace/InfiniCore/submodules/InfiniCCL/build-mars}"
JOBS="${JOBS:-$(nproc)}"

export HPCC_PATH="${HPCC_PATH:-/opt/hpcc}"
unset MACA_PATH MACA_HOME MACA_ROOT || true

mkdir -p "${PREFIX}" "${BUILD}"

cmake -S "${CCL_SRC}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DWITH_MARS=ON \
  -DWITH_HCCL=ON \
  -DWITH_METAX=OFF \
  -DWITH_MCCL=OFF \
  -DWITH_NVIDIA=OFF \
  -DWITH_NCCL=OFF \
  -DWITH_OMPI=OFF \
  -DWITH_MPICH=OFF \
  -DAUTO_DETECT_DEVICES=OFF \
  -DAUTO_DETECT_BACKENDS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF

cmake --build "${BUILD}" --parallel "${JOBS}"
cmake --install "${BUILD}"
echo "Installed InfiniCCL Mars → ${PREFIX}"
