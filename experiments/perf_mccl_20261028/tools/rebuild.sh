#!/usr/bin/env bash
set -euo pipefail
REPO=/home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise
source "$REPO/scripts/runtime_env.sh"
export INFINI_ROOT="$REPO/build/integration/metax/prefix"
export XMAKE_ROOT=y XMAKE_STATS=n XMAKE_NETWORK=n
export PYBIND11_INCLUDE=/opt/conda/lib/python3.10/site-packages/pybind11/include
export CUDA_HOME=/opt/maca CUDA_PATH=/opt/maca/tools/cu-bridge
export CPLUS_INCLUDE_PATH=/opt/conda/include/python3.10:/opt/maca/include:/opt/maca/tools/cu-bridge/include
cd "$REPO/InfiniLM"
xmake build -y -j 16 infinicore_runtime
xmake build -y -j 16 _infinilm
# Copy only linked outputs; provider libraries are unchanged.
cp build/linux/x86_64/release/libinfinicore_runtime.so "$REPO/install/lib/"
cp build/linux/x86_64/release/_infinilm*.so python/infinilm/lib/
/opt/maca/mxgpu_llvm/bin/mxcc -O3 -std=c++17 -shared -fPIC \
  csrc/infinicore/src/ops/chunk_gated_delta_rule/chunk_gated_delta_rule_metax_kernel.maca \
  -o /perf/tools/libcgdr_register.so.new
mv /perf/tools/libcgdr_register.so.new /perf/tools/libcgdr_register.so
# Build the actual provider DSO. This avoids GPU-runtime initialization inside
# the tracer caused by preloading a standalone experimental kernel library.
/opt/maca/mxgpu_llvm/bin/mxcc -O3 -std=c++17 -shared -fPIC \
 csrc/infinicore/src/ops/causal_conv1d/causal_conv1d_metax_kernel.maca \
 csrc/infinicore/src/ops/chunk_gated_delta_rule/chunk_gated_delta_rule_metax_kernel.maca \
 csrc/infinicore/src/ops/fused_gated_delta_net_gating/fused_gated_delta_net_gating_metax_kernel.maca \
 csrc/infinicore/src/ops/recurrent_gated_delta_rule/recurrent_gated_delta_rule_metax_kernel.maca \
 csrc/infinicore/src/ops/mrope/mrope_metax_kernel.maca \
 -o "$REPO/install/lib/libinfinilm_metax_extra.so.new"
mv "$REPO/install/lib/libinfinilm_metax_extra.so.new" "$REPO/install/lib/libinfinilm_metax_extra.so"
