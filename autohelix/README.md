# AutoHelix — metax-9 unified promo

Closed-loop InfiniLM vs frozen vLLM on **metax-9** (`mars` / `mars-phytium`).

| Role | Owner |
|------|--------|
| Deterministic measure (wall / profiler / facts) | InfiniOrchestrator via `scripts/promo/observe.py` |
| Iteration / accept / reject / budget | AutoHelix 0.1.1 |
| One `top_path` lever per iter | Codex (`glm-5.3`) |
| Surpass / ship | Human after `scripts/promo/stop_surpass.py` |

## Protocol

```bash
source /root/zenghua/workspace/deploy20261113/autohelix-env.sh
# activates PATH to .venv-autohelix + campaign env (SUITE_DIR absolute)

# Offline fixture (artifacts/promo/suite/compare.json):
export RUN_WALL=0 SKIP_PROFILER=1
# Live (:8000 sequential vLLM↔InfiniLM) — see scripts/promo/bringup_hint.sh
# export RUN_WALL=1 BASE_URL_VLLM=http://127.0.0.1:8000 BASE_URL_INFINILM=http://127.0.0.1:8000
# export HCTRACER_ROOT=/opt/hpcc SKIP_PROFILER=0

autohelix run -c autohelix/promo-unified-metax9.yaml -n 5
# other terminal:
autohelix watch
python scripts/promo/stop_surpass.py   # exit 0 → human ship
```

Grafana publish is deferred. Use `autohelix watch`, not Grafana.
