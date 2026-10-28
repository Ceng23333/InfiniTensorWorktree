# QY kernel snapshot: EM pending

Tag: `v2026.10.28-qy-kernel-em-pending`.
This is a source checkpoint, **not a validated production release**.

## Included state

- QY sampling/chunk-slot fixes and their regression tests.
- Opt-in fused Add/RMSNorm, RoPE/KV launch tuning, restored single-matrix GEMM
  dispatch, profiling tools, and recorded investigation reports.
- FULL-prefill captured-length guard and padding fixes. The original FULL
  configuration failed the first long-context EM wave; the guarded engine's
  four 32-token diagnostic prefixes match the retained vLLM reference.
- A new `INFINI_QY_SILU_PACKED=1` draft. It is disabled by default, has not been
  GPU-tested, and is **not present in the frozen provider used by the running EM
  gate**. Do not interpret this checkpoint as promoting that draft.

## Active verification

The full 59-sample baseline/candidate run is
`profiling/fp-ab-conc4/results/em_gate_20261028_143523`.
At checkpoint preparation, the corrected baseline had completed its first wave
cleanly at 1/4 EM. A full pass has not been established. Live summaries and raw
evidence remain local and are intentionally excluded from Git.

Frozen candidate binaries used by that run:

| Binary | SHA-256 |
|---|---|
| Guarded `_infinilm` extension | `f929f6843d39cd5a8e072959866f62902c91bbe1da57a61cd635ddb7c93b7e62` |
| Candidate `libinfinicore_runtime.so` | `d061699001b46e9c9bbf1d411e90a6adde1dffe38caf34a16d9383aeda6183f8` |
| Candidate `libinfiniops.so` | `9df8acf673b5cfe872e1d5031b4e2ff2f3ee9c7fea59725f2966f9fdea5b0caf` |

The tagged source includes the later disabled SiLU preparation, so rebuilding
the entire tag is not expected to reproduce the frozen provider hash above.
Installed baseline libraries were not overwritten.

## Lock and publication

The umbrella worktree is protected with `git worktree lock`. This prevents
pruning/removal; it is not a filesystem read-only lock. The tag fixes the source
snapshot. Subsequent source development should use another worktree or a new
explicit checkpoint, while ignored EM evidence may continue to grow here.

Submodule URLs point to the existing `Ceng23333` forks so the newly pinned source
commits can be fetched recursively. InfiniOps, InfiniLM and InfiniCore receive
the same annotated tag as the umbrella. InfiniRT/InfiniCCL receive preservation
tags at their unchanged commits; the RT feature branch has moved, so its tag is
needed to keep the original dependency reachable. Third-party dependencies
remain pinned at their existing commits.

Model weights, datasets, compiled libraries, PTI databases/exports, and live
benchmark results are not included. See `profiling/fp-ab-conc4/EM_FINDINGS.md`
for the quality issue and `ROPE_KV_PROMOTION.md` for isolated kernel evidence.
