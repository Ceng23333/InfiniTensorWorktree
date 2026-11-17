#!/usr/bin/env bash
# Build InfiniRT/Ops/CCL (WITH_METAX) + InfiniLM _infinilm for this F&P pin.
# Synced tree has no .git — do not use build_infini_stack.py git validation.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$REPO/build/integration/metax}"
PREFIX="${PREFIX:-$BUILD_ROOT/prefix}"
JOBS="${JOBS:-$(nproc)}"
OPS_CONFIG="${OPS_CONFIG:-$REPO/InfiniLM/scripts/configs/infiniops_ops_metax.json}"
IC="$REPO/InfiniCore"
IL="$REPO/InfiniLM"

export XMAKE_ROOT=y
export XMAKE_STATS=n
export XMAKE_NETWORK=n
export PYBIND11_INCLUDE="${PYBIND11_INCLUDE:-/opt/conda/lib/python3.10/site-packages/pybind11/include}"
export PYTHONUNBUFFERED=1
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export CUDA_HOME="${CUDA_HOME:-$MACA_PATH}"
export CUDA_PATH="${CUDA_PATH:-$MACA_PATH/tools/cu-bridge}"
CMAKE_BIN_DIR="${CMAKE_BIN_DIR:-/home/metax/qy/workspace/pilotdeck/_toolchain/cmake-4.3.4/bin}"
export PATH="$CMAKE_BIN_DIR:/root/.local/bin:/opt/conda/bin:$MACA_PATH/bin:$MACA_PATH/mxgpu_llvm/bin:${PATH:-}"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:/opt/conda/lib/python3.10/site-packages/torch/lib:$MACA_PATH/lib:$MACA_PATH/lib64:$MACA_PATH/ompi/lib:$MACA_PATH/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}:$MACA_PATH/include/mcr:$MACA_PATH/include/common:$MACA_PATH/include/mcsparse:$MACA_PATH/include/mcblas:$MACA_PATH/include/mcdnn:$MACA_PATH/include/mcfft:$MACA_PATH/include/mcrand:$MACA_PATH/include/mcsolver:$MACA_PATH/include:$MACA_PATH/tools/cu-bridge/include"
export C_INCLUDE_PATH="${C_INCLUDE_PATH:-}:$CPLUS_INCLUDE_PATH"
export INFINI_ROOT="$PREFIX"

mkdir -p "$BUILD_ROOT" "$PREFIX" "$REPO/logs"
exec > >(tee -a "$REPO/logs/build_metax_fp.log") 2>&1

echo "== MetaX F&P build =="
echo "REPO=$REPO"
echo "BUILD_ROOT=$BUILD_ROOT"
echo "PREFIX=$PREFIX"
echo "JOBS=$JOBS"
date -Is

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake not found (expected under $CMAKE_BIN_DIR)" >&2
  exit 1
fi
cmake --version | head -1

PYTHON="${PYTHON:-$(command -v python3)}"
echo "PYTHON=$PYTHON"

echo "== provider preflight =="
"$PYTHON" "$IC/submodules/InfiniOps/scripts/resolve_linked_ops.py" \
  --devices cpu metax \
  --ops-config "$OPS_CONFIG" \
  --output-dir "$BUILD_ROOT/provider-preflight"

echo "== InfiniRT =="
if [[ -f "$PREFIX/lib/libinfinirt.so" && -f "$PREFIX/include/infini/rt.h" ]]; then
  echo "Skipping InfiniRT rebuild (found $PREFIX/lib/libinfinirt.so)"
else
  cmake -S "$IC/submodules/InfiniRT" -B "$BUILD_ROOT/infinirt" \
    -DWITH_CPU=ON -DWITH_METAX=ON \
    -DINFINI_RT_BUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
  cmake --build "$BUILD_ROOT/infinirt" --parallel "$JOBS"
  cmake --install "$BUILD_ROOT/infinirt"
fi

echo "== InfiniOps =="
if [[ "${FORCE_REBUILD_OPS:-0}" != "1" && -f "$PREFIX/lib/libinfiniops.so" ]]; then
  echo "Skipping InfiniOps rebuild (found $PREFIX/lib/libinfiniops.so)"
else
  cmake -S "$IC/submodules/InfiniOps" -B "$BUILD_ROOT/infiniops" \
    -DWITH_CPU=ON -DWITH_METAX=ON \
    -DWITH_LINKED=ON -DWITH_TORCH=ON \
    -DAUTO_DETECT_DEVICES=OFF -DAUTO_DETECT_BACKENDS=OFF \
    -DGENERATE_PYTHON_BINDINGS=OFF \
    -DGENERATE_OPERATOR_CALL_INSTANTIATIONS=ON \
    -DPython_EXECUTABLE="$PYTHON" \
    -DINFINI_RT_ROOT="$PREFIX" \
    -DINFINI_OPS_OPS="$OPS_CONFIG" \
    -DCMAKE_C_COMPILER=/usr/bin/cc \
    -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
  cmake --build "$BUILD_ROOT/infiniops" --target infiniops --parallel "$JOBS"
  cmake --install "$BUILD_ROOT/infiniops"
fi

echo "== InfiniCCL =="
if [[ -f "$PREFIX/lib/libinfiniccl.so" ]]; then
  echo "Skipping InfiniCCL rebuild (found $PREFIX/lib/libinfiniccl.so)"
else
  cmake -S "$IC/submodules/InfiniCCL" -B "$BUILD_ROOT/infiniccl" \
    -DWITH_METAX=ON -DWITH_NCCL=OFF -DWITH_MCCL=ON \
    -DWITH_OMPI=OFF -DWITH_MPICH=OFF \
    -DAUTO_DETECT_DEVICES=OFF -DAUTO_DETECT_BACKENDS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
  cmake --build "$BUILD_ROOT/infiniccl" --parallel "$JOBS"
  cmake --install "$BUILD_ROOT/infiniccl"
fi

echo "== InfiniLM (xmake) =="
cd "$IL"
export XMAKE_STATS=n
export XMAKE_NETWORK=n
export INFINI_ROOT="${INFINI_ROOT:-$PREFIX}"
PY_INC="${PY_INC:-$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')}"
export CPLUS_INCLUDE_PATH="$PY_INC:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$PY_INC:${C_INCLUDE_PATH:-}"
echo "PY_INC=$PY_INC INFINI_ROOT=$INFINI_ROOT"
xmake f -y -cv \
  --cxflags="-DINFINICORE_HPCC_VERSION_MAJOR=3 -DINFINICORE_MACA38_FLASH_TAIL_BOOL=1 -I${PY_INC}" \
  --cxxflags="-I${PY_INC}" || \
xmake f -y -cv --cxflags="-I${PY_INC}" --cxxflags="-I${PY_INC}"
xmake build -y -j "$JOBS" infinicore_runtime
xmake install -y infinicore_runtime
xmake build -y -j "$JOBS" _infinicore
xmake install -y _infinicore
xmake build -y -j "$JOBS" _infinilm
xmake install -y _infinilm

mkdir -p "$REPO/install/lib" "$REPO/install/include"
rsync -a "$PREFIX"/ "$REPO/install/" || cp -a "$PREFIX"/. "$REPO/install/"

date -Is
echo "OK: MetaX F&P stack installed under $REPO/install (INFINI_ROOT=$PREFIX)"
printf '%s\n' 0 > "$REPO/logs/build_metax_fp.exit"
