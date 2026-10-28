#!/usr/bin/env bash
# Resolve InfiniTensorWorktree paths (this repo root).
#
# Usage:
#   source /path/to/InfiniTensorWorktree/scripts/worktree_env.sh
#   require_worktree_repos InfiniCore InfiniLM

_itw_worktree_env_main() {
  local _here
  _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export INFINI_TENSOR_WORKTREE
  INFINI_TENSOR_WORKTREE="$(cd "${_here}/.." && pwd)"
  export WORKTREE_ROOT="${INFINI_TENSOR_WORKTREE}"
  export ITW_ROOT="${INFINI_TENSOR_WORKTREE}"
}

require_worktree_repos() {
  local name path
  if [[ -z "${INFINI_TENSOR_WORKTREE:-}${WORKTREE_ROOT:-}" ]]; then
    echo "error: INFINI_TENSOR_WORKTREE unset; source scripts/worktree_env.sh first" >&2
    return 1
  fi
  local root="${INFINI_TENSOR_WORKTREE:-${WORKTREE_ROOT}}"
  if [[ ! -d "${root}" ]]; then
    echo "error: InfiniTensorWorktree missing: ${root}" >&2
    echo "  git clone --recurse-submodules <url> && git submodule update --init --recursive" >&2
    return 1
  fi
  for name in "$@"; do
    path="${root}/${name}"
    if [[ ! -d "${path}" ]]; then
      echo "error: expected InfiniTensorWorktree repo: ${path}" >&2
      echo "  git submodule update --init --recursive" >&2
      return 1
    fi
  done
  return 0
}

_itw_worktree_env_main

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
  echo "INFINI_TENSOR_WORKTREE=${INFINI_TENSOR_WORKTREE}"
  echo "WORKTREE_ROOT=${WORKTREE_ROOT}"
  echo "ITW_ROOT=${ITW_ROOT}"
  require_worktree_repos InfiniCore InfiniLM
  echo "InfiniTensorWorktree OK"
fi
