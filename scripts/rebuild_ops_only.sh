#!/bin/bash
set -euo pipefail
REPO=/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise
PREFIX=/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/prefix
IC=/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/InfiniCore
OPS_CONFIG=/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/InfiniLM/scripts/configs/infiniops_ops_metax.json
export PATH=/home/metax/qy/workspace/pilotdeck/_toolchain/cmake-4.3.4/bin:/root/.local/bin:/opt/conda/bin:/opt/maca/bin:$PATH
export LD_LIBRARY_PATH=/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/prefix/lib:/opt/conda/lib/python3.10/site-packages/torch/lib:/opt/maca/lib:/opt/maca/lib64:${LD_LIBRARY_PATH:-}
export MACA_PATH=/opt/maca CUDA_HOME=/opt/maca CUDA_PATH=/opt/maca/tools/cu-bridge
PYTHON=/opt/conda/bin/python3
JOBS=32
rm -rf /home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/infiniops
cmake -S "$IC/submodules/InfiniOps" -B "/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/infiniops" \
  -DWITH_CPU=ON -DWITH_METAX=ON \
  -DWITH_LINKED=ON -DWITH_TORCH=ON \
  -DAUTO_DETECT_DEVICES=OFF -DAUTO_DETECT_BACKENDS=OFF \
  -DGENERATE_PYTHON_BINDINGS=OFF \
  -DGENERATE_OPERATOR_CALL_INSTANTIATIONS=ON \
  -DPython_EXECUTABLE="$PYTHON" \
  -DINFINI_RT_SOURCE_DIR="$IC/submodules/InfiniRT" \
  -DINFINI_OPS_OPS="$OPS_CONFIG" \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/prefix"
cmake --build "/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/infiniops" --target infiniops --parallel $JOBS
# Prefer copying build artifact into PREFIX (install may go to /usr/local when RT is subdirectory)
cp -a /home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/infiniops/src/libinfiniops.so /home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/prefix/lib/
cmake --install "/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/infiniops" || true
cp -a /home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/infiniops/src/libinfiniops.so /home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/prefix/lib/
# restore InfiniRT headers
cmake --install /home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/infinirt || true
# copy generated instantiations header into prefix
mkdir -p /home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/prefix/include/infini
cp -a $IC/submodules/InfiniOps/generated/include/infini/operator_call_instantiations.h /home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/prefix/include/infini/
cp -a $IC/submodules/InfiniOps/generated/include/infini/*.h /home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/build/integration/metax/prefix/include/infini/ 2>/dev/null || true
echo OPS_REBUILD_OK
