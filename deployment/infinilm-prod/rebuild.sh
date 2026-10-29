#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/runtime_env.sh"
export INFINI_ROOT="$REPO/build/integration/metax/prefix"
export XMAKE_ROOT=y XMAKE_STATS=n XMAKE_NETWORK=n
export PYBIND11_INCLUDE=/opt/conda/lib/python3.10/site-packages/pybind11/include
export CUDA_HOME=/opt/maca CUDA_PATH=/opt/maca/tools/cu-bridge
export CPLUS_INCLUDE_PATH=/opt/conda/include/python3.10:/opt/maca/include:/opt/maca/tools/cu-bridge/include
cd "$REPO/InfiniLM"
# Reconfigure explicitly; xmake retains the existing platform/ABI/cxflags.
HOST_MODE="${BUILD_MODE:-release}"
xmake f -y -m "$HOST_MODE"
REBUILD_ARGS=()
if [[ "${FORCE_HOST_REBUILD:-0}" == "1" ]]; then REBUILD_ARGS+=(--rebuild); fi
xmake build -y "${REBUILD_ARGS[@]}" -j "${BUILD_JOBS:-16}" infinicore_runtime
xmake build -y -j "${BUILD_JOBS:-16}" _infinicore
xmake build -y -j "${BUILD_JOBS:-16}" _infinilm
# Stage without overwriting libraries used by a running baseline server.
HOST_STAGE="${HOST_STAGE:-$REPO/build/host-stage}"
mkdir -p "$HOST_STAGE"
cp "build/linux/x86_64/$HOST_MODE"/libinfinicore_runtime.so "$HOST_STAGE/"
cp "build/linux/x86_64/$HOST_MODE"/_infinicore*.so "$HOST_STAGE/"
cp "build/linux/x86_64/$HOST_MODE"/_infinilm*.so "$HOST_STAGE/"
if [[ "${INSTALL_HOST:-1}" == "1" ]]; then
  for dest in "$REPO/install/lib" "$INFINI_ROOT/lib" python/infinicore/lib python/infinilm/lib; do
    mkdir -p "$dest"
    cp "$HOST_STAGE/libinfinicore_runtime.so" "$dest/libinfinicore_runtime.so.new"
    mv "$dest/libinfinicore_runtime.so.new" "$dest/libinfinicore_runtime.so"
  done
  for module in _infinicore _infinilm; do
    package="${module#_}"
    for lib in "$HOST_STAGE"/"$module"*.so; do
      cp "$lib" "python/$package/lib/$(basename "$lib").new"
      mv "python/$package/lib/$(basename "$lib").new" "python/$package/lib/$(basename "$lib")"
    done
  done
fi
if [[ "${HOST_ONLY:-0}" == "1" ]]; then exit 0; fi
/opt/maca/mxgpu_llvm/bin/mxcc -O3 -std=c++17 -shared -fPIC \
  csrc/infinicore/src/ops/chunk_gated_delta_rule/chunk_gated_delta_rule_metax_kernel.maca \
  -o "$REPO/build/libcgdr_register.so".new
mv "$REPO/build/libcgdr_register.so".new "$REPO/build/libcgdr_register.so"
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
