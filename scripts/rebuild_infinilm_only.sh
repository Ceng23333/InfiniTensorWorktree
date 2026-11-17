#!/bin/bash
set -euo pipefail
REPO=/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise
PREFIX=$REPO/build/integration/metax/prefix
IL=$REPO/InfiniLM
export XMAKE_ROOT=y XMAKE_STATS=n XMAKE_NETWORK=n
export INFINI_ROOT=$PREFIX
export PYBIND11_INCLUDE=/opt/conda/lib/python3.10/site-packages/pybind11/include
PY_INC=$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')
export CPLUS_INCLUDE_PATH=$PY_INC:${CPLUS_INCLUDE_PATH:-}
export C_INCLUDE_PATH=$PY_INC:${C_INCLUDE_PATH:-}
export PATH=/home/metax/qy/workspace/pilotdeck/_toolchain/cmake-4.3.4/bin:/root/.local/bin:/opt/conda/bin:/opt/maca/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:/opt/conda/lib/python3.10/site-packages/torch/lib:/opt/maca/lib:/opt/maca/lib64:${LD_LIBRARY_PATH:-}
export MACA_PATH=/opt/maca CUDA_HOME=/opt/maca CUDA_PATH=/opt/maca/tools/cu-bridge
cd "$IL"
# force rebuild random_sample object
rm -f build/.objs/infinicore_runtime/linux/x86_64/release/csrc/infinicore/src/ops/random_sample/random_sample.cc.o || true
xmake f -y -cv --cxflags="-DINFINICORE_HPCC_VERSION_MAJOR=3 -DINFINICORE_MACA38_FLASH_TAIL_BOOL=1 -I${PY_INC}" --cxxflags="-I${PY_INC}"
xmake build -y -j 32 infinicore_runtime && xmake install -y infinicore_runtime
xmake build -y -j 32 _infinicore && xmake install -y _infinicore
xmake build -y -j 32 _infinilm && xmake install -y _infinilm
mkdir -p $REPO/install && cp -a $PREFIX/. $REPO/install/
echo INFINILM_REBUILD_OK
