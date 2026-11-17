#!/usr/bin/env bash
# InfiniRT Mars backend against HPCC toolkit. Do not enable WITH_METAX.
set -euo pipefail

RT_SRC="${RT_SRC:-/workspace/InfiniCore/submodules/InfiniRT}"
PREFIX="${INFINI_ROOT:-/workspace/InfiniLM/build/integration/mars/prefix}"
BUILD="${RT_BUILD:-/workspace/InfiniCore/submodules/InfiniRT/build-mars}"
JOBS="${JOBS:-$(nproc)}"

export HPCC_PATH="${HPCC_PATH:-/opt/hpcc}"
unset MACA_PATH MACA_HOME MACA_ROOT || true

mkdir -p "${PREFIX}" "${BUILD}"

cmake -S "${RT_SRC}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DWITH_CPU=ON \
  -DWITH_MARS=ON \
  -DWITH_METAX=OFF \
  -DAUTO_DETECT_DEVICES=OFF \
  -DINFINI_RT_BUILD_TESTING=ON

cmake --build "${BUILD}" --parallel "${JOBS}"
ctest --test-dir "${BUILD}" --output-on-failure --parallel "${JOBS}"
cmake --install "${BUILD}"
echo "Installed InfiniRT Mars → ${PREFIX}"
