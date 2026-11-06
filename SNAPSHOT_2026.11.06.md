# QY FULL P+D snapshot: keep-piecewise + always-cached vocab len

Tag: `v2026.11.06-qy-full-pd`.
Source checkpoint after the graph-policy campaign (FULL P+D keep-piecewise) and
default-on tokenizer vocab-len cache in InfiniLM.

## Included state

- `INFINI_CUDAGRAPH_POLICY=full_and_piecewise` campaign stack (FA/MOE both,
  prefill full CG, skip-tail, eager hist, fused Add+RMSNorm).
- Keep-piecewise dual-bank default (no `INFINI_PREFILL_FULL_CG_SKIP_PIECEWISE`
  required for 2B; 16a3 campaign cell defaults TP2).
- Tokenizer vocab len always cached at `LLMEngine` init (`self._vocab_size`);
  `INFINI_CACHE_TOKENIZER_VOCAB_SIZE` removed.
- VocabParallelLMHead + QY InfiniCCL NCCL path retained from
  `v2026.11.03-qy-lmhead-tp-nccl`.

## Not included / not claimed

- InfiniCore SHA change (still pinned to prior IC).
- EM/quality gate refresh for this pin.

## Lock and publication

Umbrella worktree annotated tag fixes the source snapshot.
Submodule URLs remain the `Ceng23333` forks. Nested InfiniLM shares this
annotated tag name.
