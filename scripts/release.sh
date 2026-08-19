#!/usr/bin/env bash
# Freeze InfiniTensorWorktree submodule SHAs into a release commit + annotated tag.
#
# Examples:
#   TAG=v2026.08.10 ./scripts/release.sh --from-current
#   IC_SHA=... IL_SHA=... TAG=v2026.08.10 ./scripts/release.sh
#
# Does not push. Prints push commands when done.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=worktree_env.sh
source "${SCRIPT_DIR}/worktree_env.sh"

cd "${INFINI_TENSOR_WORKTREE}"

TAG="${TAG:-v$(date -u +%Y.%m.%d)}"
PIN_ARGS=()
if [[ "${1:-}" == "--from-current" ]]; then
  PIN_ARGS=(--from-current)
elif [[ $# -gt 0 ]]; then
  echo "usage: TAG=vYYYY.MM.DD $0 [--from-current]" >&2
  echo "  or set IC_SHA IL_SHA" >&2
  exit 1
fi

if git rev-parse "${TAG}" >/dev/null 2>&1; then
  echo "error: tag already exists: ${TAG}" >&2
  exit 1
fi

mapfile -t dirty < <(git status --porcelain | awk '{print $2}')
for path in "${dirty[@]:-}"; do
  [[ -z "${path}" ]] && continue
  path="${path%/}"
  case "${path}" in
    InfiniCore|InfiniLM|.gitmodules|.gitignore|scripts|scripts/*|README.md|MANIFEST|.devcontainer|.devcontainer/*) ;;
    *)
      echo "error: unexpected dirty path before release: ${path}" >&2
      echo "  commit or stash unrelated changes first" >&2
      exit 1
      ;;
  esac
done

"${SCRIPT_DIR}/pin_worktree.sh" ${PIN_ARGS[@]+"${PIN_ARGS[@]}"}

# shellcheck source=/dev/null
source "${INFINI_TENSOR_WORKTREE}/MANIFEST"

MSG="release: freeze InfiniTensorWorktree ${TAG}

IC_SHA=${IC_SHA}
IL_SHA=${IL_SHA}
"

git add MANIFEST \
  InfiniCore \
  InfiniLM \
  .gitmodules \
  .gitignore \
  scripts \
  .devcontainer \
  README.md \
  2>/dev/null || true

git add -u -- scripts README.md MANIFEST .gitmodules .gitignore .devcontainer 2>/dev/null || true

if git diff --cached --quiet; then
  echo "error: nothing staged for release commit" >&2
  exit 1
fi

git commit -m "${MSG}"
git tag -a "${TAG}" -m "${MSG}"

ITW_SHA="$(git rev-parse HEAD)"
sed -i "s/^ITW_SHA=.*/ITW_SHA=${ITW_SHA}/" "${INFINI_TENSOR_WORKTREE}/MANIFEST"
git add MANIFEST
if ! git diff --cached --quiet; then
  git commit -m "release: record ITW_SHA ${ITW_SHA} in MANIFEST"
  git tag -d "${TAG}" >/dev/null
  git tag -a "${TAG}" -m "${MSG}"
fi

echo ""
echo "Created commit $(git rev-parse --short HEAD) and tag ${TAG}"
echo ""
echo "Push when ready:"
echo "  git push origin HEAD --tags"
echo "Consumers:"
echo "  git clone --recurse-submodules <url>"
echo "  git checkout ${TAG} && git submodule update --init --recursive"
