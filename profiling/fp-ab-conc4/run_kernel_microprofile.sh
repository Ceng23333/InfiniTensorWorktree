#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "${ROOT}/../.." && pwd)"
SUPPORT="$(cd "${BASE}/../.." && pwd)"
IMAGE="${IMAGE_TAG:-dl-vllm-minicpm5-2b-qy:v2026.09.15}"
RUN_ID="${RUN_ID:-kernel_micro_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
MODE="${PROFILE_MODE:-capture}"
[[ "$MODE" == capture || "$MODE" == kprof ]] || exit 2
RESULTS="${ROOT}/results/${RUN_ID}"
mkdir -p "$RESULTS"
exec > >(tee "$RESULTS/runner.log") 2>&1
printf 'image=%s\ngpu=%s\nmode=%s\n' "$IMAGE" "${GPU_ID:-1}" "$MODE" > "$RESULTS/environment.txt"
printf 'micro_script=%s\nrope_small_block=%s\nkv_small_block=%s\n' \
  "${MICRO_SCRIPT:-kernel_microbench.py}" "${INFINI_QY_ROPE_SMALL_BLOCK:-0}" "${INFINI_QY_KV_SMALL_BLOCK:-0}" >> "$RESULTS/environment.txt"
sha256sum "$ROOT/${MICRO_SCRIPT:-kernel_microbench.py}" > "$RESULTS/probe-sha256.txt"
sha256sum "$SUPPORT/worktrees/InfiniCore-v20261009-post-refactor-prhead-install/lib/"*.so > "$RESULTS/library-sha256.txt"
mounts=()
if [[ -n "${OPS_LIBRARY:-}" ]]; then
  mounts+=(-v "$OPS_LIBRARY:/opt/infinicore/lib/libinfiniops.so:ro")
  sha256sum "$OPS_LIBRARY" >> "$RESULTS/library-sha256.txt"
fi
if [[ -n "${RUNTIME_LIBRARY:-}" ]]; then
  mounts+=(-v "$RUNTIME_LIBRARY:/opt/infinicore/lib/libinfinicore_runtime.so:ro")
  sha256sum "$RUNTIME_LIBRARY" >> "$RESULTS/library-sha256.txt"
fi
if [[ -n "${BINDING_LIBRARY:-}" ]]; then
  mounts+=(-v "$BINDING_LIBRARY:/base/InfiniLM/python/infinicore/lib/_infinicore.cpython-312-x86_64-linux-gnu.so:ro")
  sha256sum "$BINDING_LIBRARY" >> "$RESULTS/library-sha256.txt"
fi
for engine in ${ENGINES:-infinilm vllm}; do
  [[ "$engine" == infinilm || "$engine" == vllm ]] || exit 2
  # Unique names and --rm: never replace an existing server or profiling container.
  docker run --rm --name "${RUN_ID}-${engine}" --runtime dlrt \
    -e DENGLIN_DEVICES="${GPU_ID:-1}" -e TZ=Asia/Shanghai \
    -e INFINI_QY_ROPE_SMALL_BLOCK="${INFINI_QY_ROPE_SMALL_BLOCK:-0}" \
    -e INFINI_QY_KV_SMALL_BLOCK="${INFINI_QY_KV_SMALL_BLOCK:-0}" \
    -e PYTHONPATH=/base/InfiniLM/python \
    -e LD_LIBRARY_PATH=/usr/local/dlgpu/sdk/lib:/opt/infinicore/lib:/usr/local/lib/python3.12/dist-packages/torch/lib \
    -v "${BASE}:/base:ro" -v "${RESULTS}:/results:rw" \
    -v "${SUPPORT}/worktrees/InfiniCore-v20261009-post-refactor-prhead-install/lib:/opt/infinicore/lib:ro" \
    "${mounts[@]}" \
    --entrypoint /bin/bash "$IMAGE" -c '
      set -eo pipefail
      source /usr/local/dlgpu/sdk/env.sh
      set -u
      dlsmi > "/results/$1-device.txt"
      args=(capture --activity-mask cmd,cu,curt)
      if [[ "$2" == kprof ]]; then
        args=(kprof --targets cu --kernel-name "$5")
      fi
      timeout --signal=TERM --kill-after=30s 300s dlpti_tools "${args[@]}" --cache-control none --data-file "/results/$1.db" -- \
        python3 "/base/profiling/fp-ab-conc4/$4" --engine "$1" --norm-mode "$3" --iterations 5 --output "/results/$1-checks.json" 2>&1 | tee "/results/$1-profiler.log"
      format=json
      [[ "$2" != kprof ]] || format=metric
      dlpti_tools export --format "$format" --output "/results/$1.$format" "/results/$1.db"
    ' bash "$engine" "$MODE" "${NORM_MODE:-separate}" "${MICRO_SCRIPT:-kernel_microbench.py}" \
      "${KERNEL_FILTER:-.*(AddKernel|RmsNormKernel|SiluAndMulKernel|fused_add_rms_norm|act_and_mul_kernel|ropeThreadPerItemKernel|ReshapeAndCacheFlashKernel).*}"
done
python3 "$ROOT/validate_kernel_profile.py" "$RESULTS" --mode "$MODE"
echo "Results: $RESULTS"
