#!/usr/bin/env bash
# Build InfiniRT (if needed) + InfiniOps + InfiniCCL + InfiniLM for Mars/HPCC.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${INFINI_ROOT:-/workspace/InfiniLM/build/integration/mars/prefix}"

"${SCRIPT_DIR}/build-infinirt-mars.sh"
"${SCRIPT_DIR}/build-infiniops-mars.sh"
"${SCRIPT_DIR}/build-infiniccl-mars.sh"
"${SCRIPT_DIR}/build-infinilm-mars.sh"
echo "Mars stack installed → ${PREFIX}"
