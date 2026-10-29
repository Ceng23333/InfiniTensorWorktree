#!/usr/bin/env bash
set -euo pipefail
# Keep GPU kernel DSOs out of mcTracer's own process. Preserve any profiler
# preload when adding the candidate DSO to the actual inference process.
if [[ -n "${PERF_KERNEL_PRELOAD:-}" ]]; then
  export LD_PRELOAD="$PERF_KERNEL_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
fi
exec bash /home/metax/qy/workspace/pilotdeck/InfiniTensorWorktree-v20261023-qy-minicpm5-2b-full-piecewise/scripts/run_server.sh
