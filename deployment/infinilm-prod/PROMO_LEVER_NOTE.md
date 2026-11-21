# Promo lever applied

- path_id: `host_graph_command`
- lever: reduce `NUM_BLOCKS` from 2048 to the production-known-good 512 default (start.sh + production-runtime-manifest.json)
- effect: cuts paged-KV allocation and decode graph block-table metadata by 4x while retaining 8x the roughly 64 blocks needed by the measured P64 128->128 workload; every graph, scheduler, and sampling dial remains unchanged
- evidence: InfiniProfiler ranks `host_graph_command` first, and PagedCompiler captures block tables against total `num_blocks`; the measured workload needs about one 256-token block per request
- next: run the required unprofiled wall observe; if this is flat or regressive, revert to 2048 and require source scope for Mars FA capture and batched sampling
