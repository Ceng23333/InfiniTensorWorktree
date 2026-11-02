# InfiniLM FULL CUDA-graph policy

Date: 2026-10-27. Base: EM-fixed `v2026.10.23-qy-minicpm5-2b-batch-sample-fix`
plus the opt-in FULL-prefill patch. Image/SDK: `dl-vllm-minicpm5-2b-qy:v2026.09.15`,
SDK 4.2.1. Rebuilt extension SHA-256:
`468a298240be91df2f36f5c333280781a2c7b115285e3181964ab8a8c274b8ec`.

## Policy promoted

The default production behavior is unchanged. Setting:

```text
INFINI_PREFILL_FULL_CG=1
INFINI_FA_INGRAPH=both
INFINI_MOE_INGRAPH=both
INFINI_CUDAGRAPH_POLICY=full_and_piecewise
```

captures the prefill as one whole `model_->forward()` graph, analogous to
vLLM `CUDAGraphMode::FULL`, instead of pre/post graph segments around an eager
FA call. Decode remains the existing InfiniLM FULL graph. Runtime inputs are
copied into a 512-token bucket with zeroed tails; whole-model last-token logits
are packed back into InfiniLM's expected `[1,seq,vocab]` boundary layout.

The qualification configuration was otherwise matched to the vLLM target:
8 prompts x 512 tokens, 128 requested output tokens, concurrency 4, chunked
prefill 512, BF16, flash attention, prefix cache disabled, EOS ignored.

## Clean qualification

Three clean server starts, each warmed before measurement:

| Metric | InfiniLM FULL | Prior F&P | vLLM target |
|---|---:|---:|---:|
| Median makespan | **71.961 s** | 76.608 s | 19.099 s |
| Median throughput | **14.105 tok/s** | 13.262 tok/s | 53.616 tok/s |
| Median p50 TTFT | **1610.151 ms** | 2776.499 ms | 524.836 ms |
| Output chunks | 1015 | 1016 | 1024 |

FULL improves makespan by **1.065x** over F&P and **1.043x** over eager.
It remains **3.767x slower** than the vLLM target by makespan. Empty token text
explains the 1014-1015 visible-chunk count; all requests ended with
`finish_reason=length`.

## DLPTI replay facts

Trace: `results/infinilm_full_cg_pti_20261027_1625`. Capture delay was 40 s, so
it covers warmed and measured work.

| Item | Value |
|---|---:|
| Graph launch records | 36,984 |
| Unique graph exec IDs | 156 |
| Graphs launched 726 times | 43 |
| Profiled makespan | 73.441 s |

Repeated `graph_exec_id` records verify graph replay.

## Differential profile versus prior F&P

Largest CU-kernel deltas (`FULL - F&P`):

| Kernel/bucket | F&P | FULL | Delta |
|---|---:|---:|---:|
| BF16 Copy | 1822.229 ms | 69.585 ms | **-1752.644 ms** |
| ReshapeAndCacheFlash | 714.804 ms | 1169.896 ms | +455.092 ms |
| Add | 7253.424 ms | 7130.523 ms | -122.901 ms |
| Flash attention | 7366.188 ms | 7368.235 ms | +2.047 ms |

Most of FULL's gain comes from eliminating segment-boundary copies. The dominant
attention and elementwise kernels remain essentially unchanged, explaining why
the gain is only ~6.5% rather than the vLLM-level gap.

## Remaining gap versus vLLM

The vLLM PTI trace shows substantially more efficient model kernels. Normalizing
CU+TU time by flash-call count in the two traces:

| Trace | CU+TU per flash call |
|---|---:|
| vLLM FULL | 0.709 ms |
| InfiniLM FULL | 1.274 ms |

Notable per-call differences include vLLM fused add/RMSNorm (~9.7 us) versus
InfiniLM separate Add/RMSNorm, vLLM RoPE (~9.7 us) versus InfiniLM RoPE
(~33.9 us), and InfiniLM flash at ~225.8 us versus vLLM flash at ~127.9 us.
The next work is kernel-level: fused residual+norm, QY RoPE, and attention
launch shape. Another graph-policy change alone is unlikely to close the gap.

## Artifacts

- Clean facts: `results/infinilm_full_cg_clean_20261027_1602/target-facts.json`
- PTI graph facts: `results/infinilm_full_cg_pti_20261027_1625/pti-facts.json`
- FULL profile summary: `results/infinilm_full_cg_pti_20261027_1625/full-summary.txt`
- FULL versus F&P comparison: `results/infinilm_full_cg_pti_20261027_1625/compare-fp-full.txt`
- Clean runner: `run_infinilm_full_policy.sh`
- PTI runner: `run_infinilm_full_policy_pti.sh`
