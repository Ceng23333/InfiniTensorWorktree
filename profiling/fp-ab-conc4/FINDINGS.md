# full_and_piecewise A/B findings

Run: `fp_ab_20261027_1145` (timing) and `fp_dlpti_20261027_1218` (DLPTI).

Base: EM-fixed `InfiniTensorWorktree-v20261009-post-refactor`; pins in
`base-MANIFEST.txt`: ITW `070b0a9`, InfiniCore `82dcc1f`, InfiniLM `56f9b50`.
Workload: 8 distinct 512-token prompts, 128 requested output tokens,
closed-loop concurrency 4, prefix caching disabled, EOS ignored, BF16,
flash-attn, 512-token chunked prefill, GPU 1.

## Unprofiled speed

Median of three clean restarts, each warmed before measurement:

| Metric | Eager | Full-and-piecewise | Graph / eager |
|---|---:|---:|---:|
| Median makespan | 75.081 s | 76.608 s | **0.980x** |
| Median visible-token throughput | 13.532 tok/s | 13.262 tok/s | 0.980x |
| Median p50 TTFT | 1121 ms | 2776 ms | 2.476x slower |

Per-run makespans:

| Run | Eager | F&P | Eager / F&P |
|---|---:|---:|---:|
| 1 | 73.486 s | 75.658 s | 0.971x |
| 2 | 75.979 s | 77.023 s | 0.986x |
| 3 | 75.081 s | 76.608 s | 0.980x |

All runs returned 1016 visible content chunks. EOS produces an empty final
token, so this is expected from 8 x 127 visible chunks and confirms matched
output volume.

## DLPTI attribution

Both captures used `cmd,cu,curt` and an 80-second delay. They therefore begin
after model initialization but include the tail of warmup plus the measured
workload. Use them for differential attribution, not as isolated one-pass
profiles. Profiler timing itself showed graph at 77.45 s versus eager at
78.42 s; this reverses the small unprofiled gap and indicates profiler noise
or overhead.

Timed activity totals:

| Kind | Eager | F&P | Delta |
|---|---:|---:|---:|
| CU | 13.515 s | 24.195 s | +10.680 s |
| TU | 16.616 s | 20.115 s | +3.499 s |
| MEMCPY | 2.875 s | 3.971 s | +1.096 s |
| COMMAND | 0.975 s | 0.675 s | -0.300 s |

The graph trace has substantially fewer host API records
(RUNTIME 2.00M vs 3.57M; DRIVER 1.81M vs 5.63M), but executes more timed
device work. Largest CU kernel deltas are:

| Kernel/bucket | Eager | F&P | Delta | Count |
|---|---:|---:|---:|---:|
| Add (other) | 3.792 s | 7.253 s | +3.462 s | 66,430 -> 69,036 |
| Flash attention | 4.382 s | 7.366 s | +2.984 s | 31,452 -> 32,604 |
| BF16 Copy | 0 s | 1.822 s | +1.822 s | 0 -> 12,020 |
| RoPE | 1.381 s | 2.306 s | +0.926 s | 66,432 -> 68,988 |
| SiluAndMul | 1.002 s | 1.685 s | +0.683 s | 33,215 -> 34,494 |
| RMSNorm | 1.175 s | 1.757 s | +0.582 s | 67,221 -> 69,809 |

Graph logs confirm the intended path:
`captured bucket=512 mid=false layers=42 segments=85 pre=qkv_rope`.

## Conclusion

On this reduced common workload, `full_and_piecewise` does **not** speed up the
EM-fixed 2B build. It is ~2% slower end to end and ~2.5x worse at p50 TTFT.
The likely cause is extra/more expensive device work and copies in the
piecewise path; lower host API traffic is not enough to compensate. The next
code-level target is segment-boundary Add/Copy overhead and the slower
flash-attention launches, followed by RoPE/Silu/RMSNorm count growth.

The stock DLPTI parser is safe but slow for multi-GiB exports; this result set
used `fast_dlpti_parse.py`, a line-oriented adapter reusing its aggregation
and compare logic.
