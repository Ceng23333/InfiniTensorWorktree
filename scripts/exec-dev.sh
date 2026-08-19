#!/usr/bin/env bash
# Run a command inside infinilm-dev-refactor-dev with Mars/HPCC env (no MACA alias).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${CONTAINER_NAME:-infinilm-dev-refactor-dev}"
GPU="${HPCC_VISIBLE_DEVICES:-1}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command...>" >&2
  exit 1
fi

exec docker exec \
  -e "HPCC_PATH=/opt/hpcc" \
  -e "HPCC_VISIBLE_DEVICES=${GPU}" \
  -e "INFINI_ROOT=/workspace/InfiniLM/build/integration/mars/prefix" \
  "${NAME}" bash -lc "
    set -eo pipefail
    unset MACA_PATH MACA_HOME MACA_ROOT || true
    source /workspace/scripts/hpcc-env.sh
    cd /workspace
    $*
  "
