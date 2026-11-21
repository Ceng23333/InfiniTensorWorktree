# Promo lever applied

- path_id: `host_graph_command`
- lever: restore `INFINI_MAX_NUM_BATCHED_TOKENS` from 8192 to the prior known-good 2048 default (start.sh + Dockerfile.runtime + production-runtime-manifest.json)
- effect: rolls back the iteration-3 launch-budget trial after the unprofiled wall stayed exactly flat at 0.67959; legacy graph policy, V1 scheduling, and skip-tail sync remain unchanged
- evidence: iteration-3 notes required reverting this dial if flat; pinned source confirms the persistent command overhead comes from Mars FA graph breaks and per-row Mars sampling, not the scheduler token budget
- next: deployment-only levers on `host_graph_command` are exhausted; source scope is required to validate persistent Mars FA capture and batched Mars sampling without deleting safety guards
