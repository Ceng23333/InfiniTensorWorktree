# InfiniLM FULL policy versus vLLM — matched PTI profile

**Correctness alert (2026-10-28):** the original FULL-prefill configuration failed
the first long-context EM wave. These timing results are not EM-qualified.
See [EM_FINDINGS.md](EM_FINDINGS.md) for evidence and the replay-guard investigation.

Date: 2026-10-27. GPU: KS38 device 1. SDK/DLPTI: 4.2.1.
Image: `dl-vllm-minicpm5-2b-qy:v2026.09.15`.

Both traces used the same reduced workload and capture scope:

- 8 distinct prompts, 512 prompt tokens each
- 128 requested output tokens, EOS ignored
- closed-loop concurrency 4
- chunk size / scheduler token budget 512
- prefix caching disabled
- BF16, paged flash attention
- `dlpti_tools capture --activity-mask cmd,cu,curt --delay 10000`
- one server process
- four warmups plus three measured workloads inside the trace (the client also
  warms up on each of the three measured invocations)

Follow-up (2026-10-28): see [KERNEL_DEEP_DIVE.md](KERNEL_DEEP_DIVE.md) for
measured-window launch-shape attribution and the first hardware-counter runs.
The device tables below aggregate the entire capture, not just measured windows;
flash-call normalization does not match the number of 512-token launches.

The InfiniLM arm used the opt-in FULL-prefill policy:

```text
INFINI_CUDAGRAPH_POLICY=full_and_piecewise
INFINI_PREFILL_FULL_CG=1
INFINI_FA_INGRAPH=both
INFINI_MOE_INGRAPH=both
```

The policy string remains `full_and_piecewise` for compatibility; the authoritative
`native FULL prefill CG ... mode=vllm_full` log confirms the FULL prefill branch.

## Profiled runtime

Median of the three measured workloads inside each PTI trace:

| Metric | vLLM FULL | InfiniLM FULL | InfiniLM / vLLM |
|---|---:|---:|---:|
| Makespan | 19.327 s | 73.884 s | **3.823x** |
| Visible-token throughput | 52.983 tok/s | 13.738 tok/s | 3.857x slower |
| p50 TTFT | 527.368 ms | 1619.614 ms | 3.071x slower |
| Output chunks | 3072 | 3044 | matched within empty-token skew |

For reference, the unprofiled clean medians were 19.099 s for vLLM and 71.961 s
for InfiniLM FULL, a 3.767x gap. DLPTI added about 1.2% overhead to vLLM and
about 2.7% to InfiniLM.

## Replay verification

| Item | vLLM FULL | InfiniLM FULL |
|---|---:|---:|
| PTI graph launch records | 7,494 | 79,676 |
| Unique graph exec IDs | 97 | 156 |
| Hottest replay counts | 1,777 and 1,775 | 43 graphs at 1,694 |

Both arms repeatedly replay captured graphs. InfiniLM emits more PTI graph
launch records because its graph runtime still exposes layer/device segments,
even though prefill is captured as a whole-forward graph. The important point is
that there is no eager fallback: replay counts scale with all measured steps.

## Device work normalized per flash call

Flash-call counts match closely, making this the most reliable normalization:

| Engine | Flash calls |
|---|---:|
| vLLM | 74,719 |
| InfiniLM FULL | 74,564 |

| Device class | vLLM us/flash | InfiniLM us/flash | InfiniLM / vLLM |
|---|---:|---:|---:|
| CU kernels | 199.209 | 666.811 | **3.347x** |
| TU kernels | 509.262 | 572.392 | 1.124x |
| MEMCPY | 51.449 | 66.943 | 1.301x |
| CU + TU | 708.471 | 1239.203 | **1.749x** |

TU time is similar, but InfiniLM's CU-side work is the dominant regression.

## Kernel attribution

| Operation | vLLM us/flash | InfiniLM us/flash | InfiniLM / vLLM | Extra us/flash |
|---|---:|---:|---:|---:|
| Flash attention | 127.890 | 228.592 | 1.787x | +100.702 |
| Add + RMSNorm | 20.725 | 258.268 | **12.462x** | +237.543 |
| RoPE | 10.053 | 68.228 | **6.787x** | +58.176 |
| Silu/activation | 16.097 | 49.755 | 3.091x | +33.658 |
| KV-cache reshape | 5.592 | 33.968 | 6.075x | +28.376 |

These five CU categories account for approximately **458.5 us/flash** of extra
CU time, nearly all of InfiniLM's **467.6 us/flash** CU gap. The remaining
combined CU+TU gap is **530.7 us/flash**.

## Conclusion

The FULL graph policy removes segment-boundary copies and confirms replay, but
does not close the vLLM gap. After normalization, InfiniLM remains **1.749x
slower in CU+TU device time** and about **3.8x slower end to end**.

The next optimization targets, in order, are:

1. Fuse or optimize residual Add + RMSNorm.
2. Replace or optimize QY RoPE.
3. Improve flash-attention launch shape / kernel selection.
4. Optimize activation and KV-cache reshape kernels.

Further graph-policy changes alone are unlikely to close the gap.

## Artifacts

- InfiniLM matched PTI facts: `results/infinilm_full_cg_pti_matched_20261027_1735/pti-facts.json`
- InfiniLM kernel summary: `results/infinilm_full_cg_pti_matched_20261027_1735/infinilm-summary.txt`
- Normalized comparison: `results/infinilm_full_cg_pti_matched_20261027_1735/full-vs-vllm.json`
- vLLM target facts: `results/vllm_pti_target_20261027_1520/target-facts.json`
- vLLM kernel summary: `results/vllm_pti_target_20261027_1520/vllm-summary.txt`
- Matched runner: `run_infinilm_full_policy_pti_matched.sh`
- Comparison utility: `compare_full_vs_vllm.py`
