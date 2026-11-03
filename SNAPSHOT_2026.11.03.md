# QY TP2 snapshot: lm-head vocab-parallel + InfiniCCL NCCL

Tag: `v2026.11.03-qy-lmhead-tp-nccl`.
Source checkpoint after enabling Denglin QY InfiniCCL/NCCL and VocabParallelLMHead TP2 A/B.

## Included state

- InfiniCCL `WITH_QY` + NCCL (GetUniqueId/CommInitRank/AllReduce/Send/Recv/AllGather).
- VocabParallelLMHead with root=0 Send/Recv gather; `INFINI_VOCAB_PARALLEL=0/1`.
- Build stack pins Denglin NCCL at `/home/admin01/dlearning-soft/nccl`.
- TP2 InfiniLM-only A/B: p512 o1024 decode ON 6.98 vs OFF 6.76 (~3% uplift).
  See `llm_performace_benchmark/result/20261103_162948_2cef8ec5/FINDINGS_tp2_lmhead_ab.md`.

## Not included / not claimed

- InfiniLM vs vLLM GLOBAL_MATRIX TP2 column fill.
- Default-on fused Add+RMSNorm.
- EM/quality gate for vocab-parallel or fused norm.

## Lock and publication

Umbrella worktree protected with `git worktree lock`. Tag fixes the source snapshot.
Submodule URLs remain the `Ceng23333` forks. Nested InfiniCore/InfiniLM/InfiniCCL
(and preservation tags on InfiniOps/InfiniRT) share this annotated tag name.
