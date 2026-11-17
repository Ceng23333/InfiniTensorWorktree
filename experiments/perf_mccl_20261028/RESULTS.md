# Measured iterations

Current vLLM reference: 54.601 tok/s; 2.059 s TTFT. Prefix caching off; prompt lengths match (30 and 10,718).

| Candidate | Decode tok/s | Prefill TTFT (s) | Phase / verdict |
| --- | ---: | ---: | --- |
| 01_graph_metadata_fixed | — | — | failed; incomplete response: done=False, text='', usage={}, ttft=None |
| 02_register_state_eager | 6.578 | 8.235 | failed; server exited during startup |
| 03_register_state_graph | 4.897 | 12.037 | failed; server exited during startup |
| 04_corrected_register_eager | 4.988 | 8.233 | failed; server exited during startup |
| 05_corrected_register_decode_graph | 7.753 | 8.271 | failed; server exited during startup |
| 06_corrected_integrated_eager | 5.021 | 8.272 | failed; incomplete response: done=False, text='', usage={}, ttft=None |

Throughput is completion tokens / entire request elapsed time, excluding the first warmup request. Early EOS/refusal means token counts differ; such candidates are not accepted for fixed-length decode comparisons. See raw host JSON and smoke outputs. Device graph candidates generated incorrect text and were rejected.

Kernel-only delta-rule validation: 2,048-token BF16 100.986 ms → 20.253 ms (4.99×), tested outputs identical to the original. Splitting value columns alone regressed long-sequence timing and was rejected; warp reductions added only about 9% in the microbenchmark and were not integrated.

The optimization that improved clean host prefill measured 28.765 s → 8.235 s (71.4% less). This does not establish full-model accuracy; the current vLLM reference itself fails the simple arithmetic smoke probe.

Additional isolated prefill captures 07 (default tracing) and 08 (kernel-only, 37 MB) both failed without a complete HTTP stream. Their state files explicitly record failure. The production service is restored after the experiments; see `results/production_restored.json`.
