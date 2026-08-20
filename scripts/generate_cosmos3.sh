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

# Quality defaults. Higher resolution and more inference steps usually improve
# quality but increase VRAM use and generation time.
COSMOS3_WIDTH="${COSMOS3_WIDTH:-1280}"
COSMOS3_HEIGHT="${COSMOS3_HEIGHT:-720}"
COSMOS3_STEPS="${COSMOS3_STEPS:-35}"
COSMOS3_GUIDANCE_SCALE="${COSMOS3_GUIDANCE_SCALE:-4.0}"
COSMOS3_FLOW_SHIFT="${COSMOS3_FLOW_SHIFT:-10.0}"
COSMOS3_SEED="${COSMOS3_SEED:-42}"

# Duration defaults: 81 frames / 24 FPS = 3.375 seconds.
# Prefer 4n+1 frame counts, such as 49, 61, or 81.
COSMOS3_NUM_FRAMES="${COSMOS3_NUM_FRAMES:-81}"
COSMOS3_FPS="${COSMOS3_FPS:-24}"

mkdir -p "$(dirname "$COSMOS3_OUTPUT")"

exec "$VENV_PATH/bin/python" "$ROOT_DIR/scripts/run_cosmos3.py" \
    --width "$COSMOS3_WIDTH" \
    --height "$COSMOS3_HEIGHT" \
    --steps "$COSMOS3_STEPS" \
    --guidance-scale "$COSMOS3_GUIDANCE_SCALE" \
    --flow-shift "$COSMOS3_FLOW_SHIFT" \
    --seed "$COSMOS3_SEED" \
    --num-frames "$COSMOS3_NUM_FRAMES" \
    --fps "$COSMOS3_FPS" \
    "$@"
