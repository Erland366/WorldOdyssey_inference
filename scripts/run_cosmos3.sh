#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${WORLDODYSSEY_BACKEND_VENV:-$ROOT_DIR/.venv}"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    echo "Error: the backend environment is not installed. Run ./install.sh first." >&2
    exit 1
fi

export COSMOS3_MODEL="${COSMOS3_MODEL:-nvidia/Cosmos3-Nano}"
export COSMOS3_OUTPUT="${COSMOS3_OUTPUT:-$ROOT_DIR/artifacts/backend-videos/cosmos3-nano.mp4}"
mkdir -p "$(dirname "$COSMOS3_OUTPUT")"

exec "$VENV_PATH/bin/python" "$ROOT_DIR/scripts/run_cosmos3.py" "$@"
