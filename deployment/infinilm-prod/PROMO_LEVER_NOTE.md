# Promo lever applied

- path_id: `host_graph_command`
- lever: revert `INFINI_CUDAGRAPH_POLICY` from `full_and_piecewise` to `legacy` (start.sh + Dockerfile.runtime + production-runtime-manifest.json)
- effect: removes the measured-flat F&P default while retaining `INFINI_GRAPH_SKIP_TAIL_SYNC=1`
- evidence: F&P changed the primary wall from 0.67959 to 0.67959; decode still uses the Mars-attention-split graph
- next: source scope is required to validate and allowlist Mars Qwen3-8B TP4 P64 decode FA capture; no remaining static launch knob removes the mid-segment syncs
