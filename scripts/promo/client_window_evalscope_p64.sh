#!/usr/bin/env bash
# EvalScope P64/N64 128→128 client window for attach_serve (runs where OpenAI URL is reachable).
set -euo pipefail
URL="${OPENAI_BASE_URL:-http://127.0.0.1:8200}"
MODEL="${MODEL:-Qwen3-8B}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/models/Qwen3-8B}"
if [[ ! -f "${TOKENIZER_DIR}/tokenizer_config.json" ]]; then
  TOKENIZER_DIR="/data-aisoft/mechdancer/models/Qwen3-8B"
fi
EVALSCOPE_BIN="${EVALSCOPE_BIN:-}"
if [[ -z "${EVALSCOPE_BIN}" ]]; then
  for c in \
    /root/micromamba/envs/aiperf/bin/evalscope \
    evalscope \
    python3; do
    if [[ -x "${c}" ]] || command -v "${c}" >/dev/null 2>&1; then
      EVALSCOPE_BIN="${c}"
      break
    fi
  done
fi
OUT="${CLIENT_WINDOW_OUT:-/tmp/evalscope_attach_p64}"
mkdir -p "${OUT}"
PARALLEL="${PARALLEL:-64}"
NUMBER="${NUMBER:-64}"

# Warmup
curl -sf --noproxy '*' --max-time 120 \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":8,\"temperature\":0,\"stream\":false,\"ignore_eos\":true}" \
  "${URL}/v1/chat/completions" >"${OUT}/warmup.json" || true

if [[ "${EVALSCOPE_BIN}" == *evalscope* ]] || command -v evalscope >/dev/null 2>&1; then
  BIN="${EVALSCOPE_BIN}"
  [[ "${BIN}" == python3 ]] && BIN=evalscope
  "${BIN}" perf \
    --parallel "${PARALLEL}" \
    --number "${NUMBER}" \
    --model "${MODEL}" \
    --url "${URL}/v1/chat/completions" \
    --api openai \
    --dataset random \
    --max-tokens 128 \
    --min-tokens 128 \
    --min-prompt-length 128 \
    --max-prompt-length 128 \
    --tokenizer-path "${TOKENIZER_DIR}" \
    --extra-args '{"ignore_eos": true}' \
    --no-test-connection \
    --outputs-dir "${OUT}/evalscope_p64_128_${MODEL}" \
    >"${OUT}/evalscope_console.log" 2>&1
  ec=$?
  echo "EXIT=${ec}" | tee -a "${OUT}/evalscope_console.log"
  exit "${ec}"
fi

# Fallback: concurrent OpenAI chat completions (P64 shape) without evalscope binary
python3 - <<'PY'
import json, os, urllib.request, concurrent.futures, time
url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8200").rstrip("/") + "/v1/chat/completions"
model = os.environ.get("MODEL", "Qwen3-8B")
n = int(os.environ.get("NUMBER", "64"))
# ~128 token-ish prompt
prompt = ("Explain quantum entanglement briefly. " * 20)[:500]
body = json.dumps({
  "model": model,
  "messages": [{"role": "user", "content": prompt}],
  "max_tokens": 128,
  "temperature": 0,
  "stream": False,
  "ignore_eos": True,
}).encode()

def one(i):
  req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
  with urllib.request.urlopen(req, timeout=600) as r:
    return r.status

t0 = time.time()
ok = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
  for st in ex.map(one, range(n)):
    if st == 200:
      ok += 1
print(f"fallback_p64 ok={ok}/{n} elapsed={time.time()-t0:.1f}s", flush=True)
raise SystemExit(0 if ok == n else 1)
PY
