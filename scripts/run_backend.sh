#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${WORLDODYSSEY_BACKEND_VENV:-$ROOT_DIR/.venv}"
HOST="${WORLDODYSSEY_BACKEND_HOST:-127.0.0.1}"
PORT="${WORLDODYSSEY_BACKEND_PORT:-8000}"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    echo "Error: the backend environment is not installed. Run ./install.sh first." >&2
    exit 1
fi

export WORLDODYSSEY_SGLANG_BASE_URL="${WORLDODYSSEY_SGLANG_BASE_URL:-http://127.0.0.1:30000}"
export WORLDODYSSEY_SGLANG_VIDEO_API_FORMAT="${WORLDODYSSEY_SGLANG_VIDEO_API_FORMAT:-multipart}"

source "$VENV_PATH/bin/activate"
exec python "$ROOT_DIR/scripts/serve_video_backend.py" \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
