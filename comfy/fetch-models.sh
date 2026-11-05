#!/usr/bin/env bash
# Fetches the weights in models.txt into MODELS_DIR; resumes partial downloads.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
MODELS_DIR=${MODELS_DIR:-/mnt/data/models/comfy}
grep -v '^#' "$here/models.txt" | while read -r dir url; do
  mkdir -p "$MODELS_DIR/$dir"
  out="$MODELS_DIR/$dir/$(basename "$url")"
  curl -fL --retry 5 -C - -o "$out" "$url"
done
