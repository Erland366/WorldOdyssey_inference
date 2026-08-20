#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${WORLDODYSSEY_BACKEND_VENV:-$ROOT_DIR/.venv}"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    echo "Error: the backend environment is not installed. Run ./install.sh first." >&2
    exit 1
fi

export FASTWAN_OUTPUT="${FASTWAN_OUTPUT:-$ROOT_DIR/artifacts/backend-videos/fastwan-sglang-quality.mp4}"

# Quality defaults. Higher resolution and more inference steps usually improve
# quality but increase VRAM use and generation time.
FASTWAN_WIDTH="${FASTWAN_WIDTH:-832}"
FASTWAN_HEIGHT="${FASTWAN_HEIGHT:-448}"
FASTWAN_STEPS="${FASTWAN_STEPS:-3}"
FASTWAN_SEED="${FASTWAN_SEED:-1}"

# Duration defaults: 61 frames / 16 FPS = 3.8125 seconds.
# Prefer 4n+1 frame counts, such as 49, 61, or 81.
FASTWAN_NUM_FRAMES="${FASTWAN_NUM_FRAMES:-61}"
FASTWAN_FPS="${FASTWAN_FPS:-16}"

mkdir -p "$(dirname "$FASTWAN_OUTPUT")"

exec "$VENV_PATH/bin/python" "$ROOT_DIR/scripts/run_fastwan_quality.py" \
    --width "$FASTWAN_WIDTH" \
    --height "$FASTWAN_HEIGHT" \
    --steps "$FASTWAN_STEPS" \
    --seed "$FASTWAN_SEED" \
    --num-frames "$FASTWAN_NUM_FRAMES" \
    --fps "$FASTWAN_FPS" \
    "$@"
