#!/usr/bin/env bash
set -euo pipefail
NAME="${CONTAINER_NAME:-infinilm-dev-refactor-dev}"
docker exec "${NAME}" bash -lc 'pkill -f infinilm/server/inference_server.py || true'
echo "stopped inference_server in ${NAME}"
