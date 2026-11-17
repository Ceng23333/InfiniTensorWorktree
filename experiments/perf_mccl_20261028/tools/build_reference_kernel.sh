#!/usr/bin/env bash
set -euo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
/opt/maca/mxgpu_llvm/bin/mxcc -O3 -std=c++17 -shared -fPIC \
 "$TOOLS/reference/cgdr/chunk_gated_delta_rule_metax_kernel.maca" -o "$TOOLS/libcgdr_baseline.so"
