#!/usr/bin/env bash
# Runs ComfyUI in the foreground on 127.0.0.1:8188. Under `ws submit` the container carries the
# job's label, so a stopped job can't leave it holding VRAM, and runs on the job's cores with its
# memory capped at the job's declaration (`--mem`; measured peak 31.7 GB, AGENTS.md).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
DATA=${FORGE_DATA:-/mnt/data/sprite-forge}
mkdir -p "$DATA/out" "$DATA/in" "$DATA/miopen"
exec docker run --rm --name "${FORGE_CONTAINER:-sprite-forge-comfy}" \
  ${WS_JOB_LABEL:+--label "$WS_JOB_LABEL"} \
  ${WS_JOB_CPUS:+--cpuset-cpus "$WS_JOB_CPUS"} \
  $( [ "${WS_JOB_MEM_MIB:-0}" -gt 0 ] && echo "--memory ${WS_JOB_MEM_MIB}m --memory-swap ${WS_JOB_MEM_MIB}m" ) \
  --device /dev/kfd --device /dev/dri --group-add 44 --group-add 992 \
  --security-opt seccomp=unconfined --ipc host --user "$(id -u):$(id -g)" \
  -p 127.0.0.1:${FORGE_PORT:-8188}:8188 \
  -e ROCR_VISIBLE_DEVICES=0 -e MIOPEN_USER_DB_PATH=/miopen -e MIOPEN_CUSTOM_CACHE_DIR=/miopen -e HOME=/tmp \
  -v "$DATA/ComfyUI:/comfy" -v "$DATA/out:/out" -v "$DATA/in:/in" -v "$DATA/miopen:/miopen" \
  -v /mnt/data/models/comfy:/models:ro -v "$here/extra_model_paths.yaml:/extra.yaml:ro" \
  -w /comfy sprite-forge-comfy:latest \
  python3 main.py --listen 0.0.0.0 --port 8188 --output-directory /out --input-directory /in \
    --extra-model-paths-config /extra.yaml --disable-auto-launch "$@"
