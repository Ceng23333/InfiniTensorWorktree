#!/usr/bin/env bash
set -euo pipefail
image=pilotdeck-infinilm:promoted-tool-fix-20261029
name=pilotdeck-infinilm-prod
if docker container inspect "$name" >/dev/null 2>&1; then
  expected=$(docker image inspect "$image" --format '{{.Id}}')
  actual=$(docker container inspect "$name" --format '{{.Image}}')
  if [[ "$expected" != "$actual" ]]; then
    echo 'Existing production container uses another image; preserve it before replacement.' >&2
    exit 1
  fi
  docker start "$name"
else
  docker run -d --name "$name" --restart unless-stopped \
    --privileged --device /dev/mxcd --device /dev/dri \
    --security-opt label=disable --shm-size 8g \
    --network pilotdeck_default -p 8000:8000 \
    --log-opt max-size=10m --log-opt max-file=3 \
    -v /home/metax/Downloads/Qwen3.8-27B:/models/Qwen3.8-27B:ro \
    -e INFINICORE_GRAPH_DEBUG=1 -e INFINI_DECODE_GRAPH_DIAGNOSTICS=1 \
    "$image"
fi
