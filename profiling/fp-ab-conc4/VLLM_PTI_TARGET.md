# vLLM PTI target facts

Date: 2026-10-27. Host GPU: KS38, device 1. SDK/DLPTI: 4.2.1.
Image: `dl-vllm-minicpm5-2b-qy:v2026.09.15`; vLLM: `0.21.0`.

This is the vLLM target for the reduced graph workload. It uses the same
eight tokenizer-exact 512-token prompts as the InfiniLM A/B, requests 128
tokens, runs closed-loop concurrency 4, disables prefix caching, and ignores
EOS. Each run therefore returns exactly 1024 visible content chunks.

## Server configuration

```text
V1 engine
enforce_eager=false
chunked_prefill=true
max_num_seqs=4
max_num_batched_tokens=512
prefix_caching=false
```

The Denglin plugin was active and selected:

```text
cudagraph_mode=FULL
prefill capture sizes=[8,16,32,48,64,128,192,256,320,384,448,512]
decode capture sizes=[1,2,3,4]
compile range=(1,512)
```

The graph run log records 12 prefill FULL captures and 4 decode FULL captures.
Graph capture finished in 23 s and used 0.08 GiB.

## Clean target

No DLPTI wrapper. Median of three runs:

| Metric | Target |
|---|---:|
| Makespan | **19.099 s** |
| Visible-token throughput | **53.616 tok/s** |
| p50 TTFT | **524.836 ms** |
| Median request latency | 9547.911 ms |
| Output chunks | 1024 |

Per-run makespans: 19.078 s, 19.099 s, 19.113 s.

Relative to the EM-fixed InfiniLM medians:

| Comparison | Speedup |
|---|---:|
| vLLM clean versus InfiniLM eager | **3.931x** |
| vLLM clean versus InfiniLM full-and-piecewise | **4.011x** |
| TTFT ratio, InfiniLM eager / vLLM | 2.137x |
| TTFT ratio, InfiniLM F&P / vLLM | 5.290x |

## DLPTI verification

Separate run under `dlpti_tools capture` with the same workload. The trace began
after a 10 s delay, so it includes compilation/capture, warmup, and measured
runs. Profiled median makespan was 19.327 s and p50 TTFT was 527.368 ms,
consistent with the clean target.

Native export facts:

| Item | Value |
|---|---:|
| `GRAPH` launch records (`kind=32`) | 7,494 |
| Unique graph exec IDs | 97 |
| Exec IDs launched exactly 42 times | 91 |
| Hottest exec IDs | 1,777 and 1,775 launches |

Repeated `graph_exec_id` records prove graph replay rather than merely graph
capture. The 42-launch population is consistent with per-layer/partitioned
graphs in the 42-layer model. The two hot IDs correspond to repeatedly
replayed high-frequency scheduler paths.

## Artifact paths

- Clean target facts: `results/vllm_clean_target_20261027_1534/target-facts.json`
- PTI target facts: `results/vllm_pti_target_20261027_1520/pti-facts.json`
- PTI measured facts: `results/vllm_pti_target_20261027_1520/target-facts.json`
- vLLM graph/capture log: `results/vllm_pti_target_20261027_1520/vllm.final.log`
- Runner: `run_vllm_pti_facts.sh`
- Fact extractor: `extract_pti_facts.py`
