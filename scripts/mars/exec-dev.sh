#!/usr/bin/env bash
# Run a command inside the Mars InfiniLM dev container (no MACA alias).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${CONTAINER_NAME:-infinilm-dev-ar-split-mars}"
GPU="${HPCC_VISIBLE_DEVICES:-0,1,2,3}"
INFINI_ROOT_IN="${INFINI_ROOT:-/workspace/prefix-mars}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command...>" >&2
  exit 1
fi

exec docker exec \
  -e "HPCC_PATH=/opt/hpcc" \
  -e "HPCC_VISIBLE_DEVICES=${GPU}" \
  -e "INFINI_ROOT=${INFINI_ROOT_IN}" \
  "${NAME}" bash -lc "
    set -eo pipefail
    unset MACA_PATH MACA_HOME MACA_ROOT CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH || true
    source /workspace/scripts/mars/hpcc-env.sh
    export INFINI_ROOT='${INFINI_ROOT_IN}'
    export LD_LIBRARY_PATH=\"\${INFINI_ROOT}/lib64:\${INFINI_ROOT}/lib:/workspace/InfiniLM/build/linux/arm64/release:\${LD_LIBRARY_PATH:-}\"
    cd /workspace
    $*
  "
