# Promo lever applied

- path_id: `host_graph_command`
- lever: raise `INFINI_MAX_NUM_BATCHED_TOKENS` from 2048 to 8192 (start.sh + Dockerfile.runtime + production-runtime-manifest.json)
- effect: packs the measured P64 x 128-token prompt cohort into one launch budget instead of four prefill chunks, reducing chunk-boundary host work while keeping legacy graph policy and skip-tail sync unchanged
- evidence: fresh hcTracer ranks `runtime.host_graph_command` (runtime 7.198x, command 6.390x, gap 9.781x); prior F&P was exactly flat and the request asks for a distinct launch-batching lever
- next: unprofiled wall observe decides acceptance; source scope is still required to remove the 16 Mars attention host breaks per decode graph
