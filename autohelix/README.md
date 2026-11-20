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
export INFINI_ORCHESTRATOR=/root/zenghua/workspace/InfiniOrchestrator
export INFINI_TENSOR_WORKTREE=/root/zenghua/workspace/deploy20261113/InfiniTensorWorktree
export HOST_ROLE=mars DEVICE_TRACK=mars HOST_ID=metax-9 MODEL=Qwen3-8B

# Offline fixture:
export RUN_WALL=0 SKIP_PROFILER=1
# Live:
# export RUN_WALL=1
# export BASE_URL_VLLM=http://127.0.0.1:8000
# export BASE_URL_INFINILM=http://127.0.0.1:8000
# export HCTRACER_ROOT=/opt/hpcc

cd "$INFINI_TENSOR_WORKTREE"
autohelix run -c autohelix/promo-unified-metax9.yaml -n 5
# other terminal:
autohelix watch
```

Grafana publish is deferred. Use `autohelix watch`, not Grafana.
