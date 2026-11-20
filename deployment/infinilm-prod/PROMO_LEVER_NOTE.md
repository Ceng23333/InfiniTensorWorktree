# Promo lever applied

- path_id: `host_graph_command`
- lever: `INFINI_CUDAGRAPH_POLICY=full_and_piecewise` (start.sh + Dockerfile.runtime + production-runtime-manifest.json)
- effect: selects the product full/piecewise graph policy while retaining `INFINI_GRAPH_SKIP_TAIL_SYNC=1`
- prior wall: ~0.680 tok/s ratio vs frozen vLLM after skip-tail sync (still behind)
- next: source allowlist for mars FA-in-graph remains the likely true fix if this policy is flat or regressive
