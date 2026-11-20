# Agent rules — AutoHelix promo on metax-9

You are the Codex agent inside AutoHelix. AutoHelix owns accept/reject/budget.

## Each iteration

1. Read `artifacts/promo/suite/promotion_backlog.json` (or `artifacts/promo/suite/profile/<run_id>/promotion_backlog.json` / wall-only backlog under the suite). Prefer the path printed by the last observe turn.
2. Apply **exactly one** `top_path` lever.
3. Scope: only modify files under `deployment/infinilm-prod/`.
4. Do **not** edit `scripts/promo/`, `autohelix/`, harness cases, or invent `WALL_DECODE_RATIO`.
5. Keep the frozen vLLM baseline unchanged; rebuild/restart InfiniLM candidate only when the lever requires it.
6. Write AutoHelix notes / commit_summary as required by the framework.

## Do not

- Surpass/ship judgment (matrix, ITW tag, IO case `worktree` bump) — human after `scripts/promo/stop_surpass.py` exits 0
- Multi-lever shotgun changes in one iteration
- Touch Grafana or invent metrics

## Cases

- Candidate: `qwen3-8b-mars-inf--unified-tp4`
- Frozen baseline: `qwen3-8b-mars-vllm--unified-tp4`
- Host: metax-9 · track: mars / mars-phytium · model: Qwen3-8B
