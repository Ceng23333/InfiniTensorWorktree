#!/usr/bin/env bash
# pip-install InfiniLM against the Mars INFINI_ROOT prefix (xmake extensions).
set -euo pipefail

export HPCC_PATH="${HPCC_PATH:-/opt/hpcc}"
unset MACA_PATH MACA_HOME MACA_ROOT || true
export INFINI_ROOT="${INFINI_ROOT:-/workspace/InfiniLM/build/integration/mars/prefix}"
export XMAKE_ROOT="${XMAKE_ROOT:-y}"

if ! command -v xmake >/dev/null 2>&1; then
  echo "error: xmake is not on PATH; seed the xmake binary first" >&2
  exit 1
fi

cd /workspace/InfiniLM
xmake g --yes >/dev/null 2>&1 || true
xmake f -y || true
python3 -m pip install janus
python3 -m pip install . --no-build-isolation --no-deps
echo "Installed InfiniLM (Mars) with INFINI_ROOT=${INFINI_ROOT}"
