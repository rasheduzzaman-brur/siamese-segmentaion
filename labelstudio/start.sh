#!/usr/bin/env bash
# Start Label Studio with local file serving rooted at the repo's data/ dir,
# so tasks from labelstudio/tasks.py can show data/prepared/images/... and
# data/prepared/reference/... without uploading anything.
#
#   labelstudio/start.sh [--allow-local-ml] [--port 8080] [other label-studio start args]
#
# --allow-local-ml lets Label Studio call an ML backend on localhost / a
# private address (labelstudio/ml_backend.py). It turns off Label Studio's
# SSRF protection, so only use it on a machine you control.
#
# Uses the `labelstudio` conda env if it exists (Label Studio is kept out of
# the training env because of its Django dependency pins).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="${LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT:-$REPO/data}"
export LABEL_STUDIO_BASE_DATA_DIR="${LABEL_STUDIO_BASE_DATA_DIR:-$REPO/data/label-studio}"

if [[ "${1:-}" == "--allow-local-ml" ]]; then
  export SSRF_PROTECTION_ENABLED=false ML_BLOCK_LOCAL_IP=false
  shift
fi

LS=label-studio
if command -v conda >/dev/null && conda env list | grep -q '^labelstudio '; then
  LS="conda run --no-capture-output -n labelstudio label-studio"
fi
exec $LS start "$@"
