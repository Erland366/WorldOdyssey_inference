#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash examples/generate_cosmos3_i2v.sh <input-image> [extra generation options]" >&2
    echo "Example: bash examples/generate_cosmos3_i2v.sh images/planet.png --seed 123" >&2
    exit 2
fi

IMAGE_PATH="$1"
shift

if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "Input image not found: $IMAGE_PATH" >&2
    exit 2
fi

PROMPT="${COSMOS3_I2V_PROMPT:-Pick the Book}"
OUTPUT="${COSMOS3_I2V_OUTPUT:-$ROOT_DIR/artifacts/backend-videos/cosmos3-i2v.mp4}"

# Quality defaults. Increase resolution or steps for higher quality at the cost
# of more VRAM and generation time.
WIDTH="${COSMOS3_I2V_WIDTH:-480}"
HEIGHT="${COSMOS3_I2V_HEIGHT:-480}"
STEPS="${COSMOS3_I2V_STEPS:-50}"
GUIDANCE_SCALE="${COSMOS3_I2V_GUIDANCE_SCALE:-4.0}"
FLOW_SHIFT="${COSMOS3_I2V_FLOW_SHIFT:-10.0}"
SEED="${COSMOS3_I2V_SEED:-42}"

# Duration defaults: 81 frames / 24 FPS = 3.375 seconds.
# Prefer 4n+1 frame counts, such as 49, 61, or 81.
NUM_FRAMES="${COSMOS3_I2V_NUM_FRAMES:-81}"
FPS="${COSMOS3_I2V_FPS:-24}"

exec bash "$ROOT_DIR/scripts/generate_cosmos3.sh" \
    --image-path "$IMAGE_PATH" \
    --prompt "$PROMPT" \
    --output "$OUTPUT" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --steps "$STEPS" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --flow-shift "$FLOW_SHIFT" \
    --seed "$SEED" \
    --num-frames "$NUM_FRAMES" \
    --fps "$FPS" \
    "$@"
