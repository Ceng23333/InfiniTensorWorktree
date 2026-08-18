#!/usr/bin/env bash
# Checkout pinned SHAs in InfiniTensorWorktree submodules and stage gitlinks.
#
# Env (optional unless --from-current):
#   IC_SHA IL_SHA
# Or:
#   ./pin_worktree.sh --from-current   # use each submodule HEAD
#
# Writes MANIFEST and stages gitlinks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=worktree_env.sh
source "${SCRIPT_DIR}/worktree_env.sh"

FROM_CURRENT=0
if [[ "${1:-}" == "--from-current" ]]; then
  FROM_CURRENT=1
fi

IC_URL="${IC_URL:-https://github.com/Ceng23333/InfiniCore.git}"
IL_URL="${IL_URL:-https://github.com/Ceng23333/InfiniLM.git}"

require_worktree_repos InfiniCore InfiniLM

pin_one() {
  local name="$1" sha="$2"
  local path="${INFINI_TENSOR_WORKTREE}/${name}"
  echo "Pinning ${name} → ${sha} ..."
  git -C "${path}" fetch --all --tags
  git -C "${path}" checkout --detach "${sha}"
  if [[ -f "${path}/.gitmodules" ]]; then
    git -C "${path}" submodule update --init --recursive || true
  fi
}

if [[ "${FROM_CURRENT}" -eq 1 ]]; then
  IC_SHA="$(git -C "${INFINI_TENSOR_WORKTREE}/InfiniCore" rev-parse HEAD)"
  IL_SHA="$(git -C "${INFINI_TENSOR_WORKTREE}/InfiniLM" rev-parse HEAD)"
else
  : "${IC_SHA:?set IC_SHA or pass --from-current}"
  : "${IL_SHA:?set IL_SHA or pass --from-current}"
  pin_one InfiniCore "${IC_SHA}"
  pin_one InfiniLM "${IL_SHA}"
fi

IC_SHA="$(git -C "${INFINI_TENSOR_WORKTREE}/InfiniCore" rev-parse HEAD)"
IL_SHA="$(git -C "${INFINI_TENSOR_WORKTREE}/InfiniLM" rev-parse HEAD)"
ITW_SHA="$(git -C "${INFINI_TENSOR_WORKTREE}" rev-parse --verify HEAD 2>/dev/null)" || ITW_SHA=unknown
PIN_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

MANIFEST="${INFINI_TENSOR_WORKTREE}/MANIFEST"
cat > "${MANIFEST}" <<EOF
ITW_SHA=${ITW_SHA}
IC_SHA=${IC_SHA}
IL_SHA=${IL_SHA}
IC_URL=${IC_URL}
IL_URL=${IL_URL}
PIN_DATE=${PIN_DATE}
EOF

cd "${INFINI_TENSOR_WORKTREE}"
git add \
  InfiniCore \
  InfiniLM \
  MANIFEST
if [[ -f .gitmodules ]]; then
  git add .gitmodules
fi

echo ""
echo "MANIFEST:"
cat "${MANIFEST}"
echo ""
echo "Staged gitlinks. Review with: git status && git diff --cached --submodule"
