#!/usr/bin/env bash
set -euo pipefail
STACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPS="$STACK_ROOT/InfiniCore/submodules/InfiniOps"
PATCH="$STACK_ROOT/experiments/perf_mccl_20261028/infiniops-compatibility.patch"
if git -C "$OPS" apply --check "$PATCH"; then
  git -C "$OPS" apply "$PATCH"
elif git -C "$OPS" apply --reverse --check "$PATCH"; then
  echo 'InfiniOps compatibility overlay is already applied.'
else
  echo 'InfiniOps does not match the recorded baseline/overlay; see MANIFEST.' >&2
  exit 1
fi
