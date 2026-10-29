# Full 59-sample repaired-replay benchmark

Date: 2026-10-28, Asia/Shanghai. Run: `em_gate_20261028_201439`.
Started 20:14:59; completed 22:14:37 (1h 59m 38s including load and warmup).

## Outcome

All 59 samples and 15 waves completed with clean exits. **EM is 14/59
(23.73%)**, below the retained October 26 graph-enabled result of 16/59.
This is a successful execution, **not an EM qualification pass**. Optional
fused-kernel promotion remains held; the root cause of answer changes has not
been isolated by this single-arm run.

| Metric | Repaired FULL, Oct 28 | Historical graph, Oct 26 | Historical eager, Oct 26 |
|---|---:|---:|---:|
| EM | 14/59 (23.73%) | 16/59 (27.12%) | 17/59 (28.81%) |
| Aggregate output throughput | 8.585 tok/s | 7.877 tok/s | 8.093 tok/s |
| Mean request TTFT | 66.003 s | 72.715 s | 77.428 s |
| Mean request latency | 164.632 s | 185.955 s | 184.017 s |
| Measured wave time, summed | 3141.366 s | 3537.159 s | 3445.355 s |
| Generated output tokens | 26,970 | 27,862 | 27,882 |

Versus the historical graph run: throughput +8.99%, mean TTFT -9.23%, and
mean request latency -11.47%. These are historical workload comparisons, not
a fresh paired speedup measurement: runtime configurations differ, generated
output volume is 3.20% lower, and historical metrics are rounded. There is no
fresh vLLM run in this result.

Current request percentiles (linear interpolation across 59 requests):

| Metric | p50 | p95 |
|---|---:|---:|
| TTFT | 78.629 s | 109.071 s |
| Request latency | 196.864 s | 225.792 s |

Throughput is total output tokens divided by summed measured wave makespans,
not the arithmetic mean of per-wave throughputs. Inference timing excludes
model loading, graph capture and warmup. Each wave starts a fresh engine and
runs a full warmup before the measured pass. The full wall time includes both.

## Configuration and Integrity

- BF16 MiniCPM5-2B, TP1, KS38 GPU 1; container runtime `dlrt`.
- Image `dl-vllm-minicpm5-2b-qy:v2026.09.15`.
- 59 `short_easy_cot` samples; 14 four-request waves and one three-request wave.
- Greedy decoding, EOS respected, maximum 512 output tokens.
- Existing head/tail prompt truncation to 7680 tokens; all 59 prompts truncated.
- Chunk size/token budget 512, FULL prefill and graph-enabled decode, prefix
  caching disabled. Capture logs report max Q=512, max KV=8448.
- Optional fused Add/RMSNorm, small-block RoPE and small-block KV flags off.
  No new SiLU promotion. No eager fallback arm was run.
- Detailed graph/rank-worker profiling off for performance measurement.
  Completed wave logs contain no fallback/error messages; the earlier targeted
  probes provide the detailed device-only replay evidence.
- Loaded-library hashes were identical across all 15 waves and match the
  installed repair. Engine/runtime/binding snapshots, container settings,
  source diffs, GPU survey and dataset/scorer hashes are retained with the run.
- Actual concurrency is recorded in each case and its request list. The legacy
  adapter's `effective_config.concurrency=1` field is hardcoded and should not
  be interpreted as the wave concurrency.

## Answer Changes

Compared with historical graph-enabled answers: 51/59 predictions agree,
four previously correct answers became incorrect, and two became correct.
Sample IDs, answers, and prompt lengths/truncation match. The historical
summary lacks per-input hashes, so exact historical input identity is not
asserted from that file alone.

| Lost-correct sample | Expected | Current prediction | Output tokens |
|---|---|---|---:|
| `66f6a14dbb02136c067c2453` | D | A | 512 |
| `67062fbfbb02136c067cf66a` | D | A | 222 |
| `66ec3a46821e116aacb1c49f` | A | C | 512 |
| `66ebda775a08c7b9b35e0dcb` | C | A | 301 |

Gained-correct IDs: `67078ecabb02136c067d0b89`, `66ebc95f5a08c7b9b35df497`.

Two outputs have no extractable option; both contain readable reasoning and
hit the 512-token budget before producing an answer. Overall, 44/59 outputs
reach that budget. The unchanged legacy scorer uppercases the entire output
and selects the last isolated A/B/C/D, including letters in reasoning and the
English article "a". For example, one lost-correct completion says D is correct
before discussing other choices, but the scorer returns A. This is a scoring
limitation, not grounds to relabel the measured EM or dismiss the regression.
Other inspected outputs include explicit incorrect answers. No numerical
equivalence or root-cause claim follows from this benchmark.

## Artifacts and Reproduction

Run directory: `results/em_gate_20261028_201439/`.

- `em-performance.json`: validated aggregate metrics and historical comparison.
- `baseline/summary.json`: all wave statuses, scores, sample IDs and input hashes.
- `baseline/decoded-predictions.jsonl`: all decoded completions with unchanged scores.
- `baseline/infinilm/`: raw token IDs and per-request timing.
- `baseline/maps/`: loaded-library hashes for every wave.
- `baseline/logs/`: adapter stdout/stderr, including capture capacities.
- `snapshot/`, `container-inspect.json`, `gpu-start.txt`, source patches: provenance.

Historical source:
`/home/qinyiqun/llm_performace_benchmark/result/longbench_task_merged_20261026_102006/summary.json`.

```bash
USE_INSTALLED=1 EM_ARMS=baseline LIMIT=59 REPLAY_PROFILE=0 \
  bash profiling/fp-ab-conc4/run_em_gate.sh
python3 profiling/fp-ab-conc4/summarize_em_performance.py \
  profiling/fp-ab-conc4/results/em_gate_20261028_201439/baseline \
  --reference /home/qinyiqun/llm_performace_benchmark/result/longbench_task_merged_20261026_102006/summary.json
```

The reproduction benchmark creates a new dated directory; point the summarizer
at that directory for a new run. No source commit, tag update, or provider
promotion was made for this benchmark.
