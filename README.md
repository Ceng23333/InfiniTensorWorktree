# InfiniTensorWorktree

Lightweight pin umbrella for **InfiniCore**, **InfiniLM**, and **InfiniMetadata**. Annotated release tags (`vYYYY.MM.DD`) lock the deployment commit (submodule SHAs + `MANIFEST`).

## Layout

```text
InfiniTensorWorktree/
  InfiniCore/       # submodule
  InfiniLM/         # submodule
  InfiniMetadata/   # submodule
  MANIFEST          # ITW_SHA, IC_SHA, IL_SHA, IM_SHA
  scripts/          # worktree_env, pin_worktree, release
```

## Clone

```bash
git clone --recurse-submodules https://github.com/Ceng23333/InfiniTensorWorktree.git
cd InfiniTensorWorktree
git checkout vYYYY.MM.DD
git submodule update --init --recursive
source scripts/worktree_env.sh
```

Sibling layout with InfiniOrchestrator (default):

```text
workspace/
  InfiniOrchestrator/
  InfiniTensorWorktree/   # this repo
```

InfiniOrchestrator resolves `INFINI_TENSOR_WORKTREE` to `../InfiniTensorWorktree` unless overridden.

## Release pin

```bash
TAG=vYYYY.MM.DD ./scripts/release.sh --from-current
# or:
IC_SHA=... IL_SHA=... IM_SHA=... TAG=vYYYY.MM.DD ./scripts/release.sh
git push origin HEAD --tags
```

Manifest fields: `ITW_SHA`, `IC_SHA`, `IL_SHA`, `IM_SHA`, URLs, `PIN_DATE`.
