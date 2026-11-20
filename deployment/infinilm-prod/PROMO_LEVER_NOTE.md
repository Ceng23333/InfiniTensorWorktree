# Promo lever applied

- path_id: `host_graph_command`
- lever: `INFINI_GRAPH_SKIP_TAIL_SYNC=1` (start.sh + production-runtime-manifest.json)
- effect: removes Graph::run **tail** syncStream only; per-layer mid-segment syncs remain while mars FA is capture-unsafe
- wall: ~0.635 → ~0.680 tok/s ratio vs frozen vLLM (still behind)
- next: source allowlist for mars FA-in-graph (not an env knob)
