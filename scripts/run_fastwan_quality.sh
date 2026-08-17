#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${WORLDODYSSEY_BACKEND_VENV:-$ROOT_DIR/.venv}"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    echo "Error: the backend environment is not installed. Run ./install.sh first." >&2
    exit 1
fi

export FASTWAN_OUTPUT="${FASTWAN_OUTPUT:-$ROOT_DIR/artifacts/backend-videos/fastwan-sglang-quality.mp4}"
mkdir -p "$(dirname "$FASTWAN_OUTPUT")"

exec "$VENV_PATH/bin/python" "$ROOT_DIR/scripts/run_fastwan_quality.py" "$@"
