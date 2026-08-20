#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${HUNYUAN_MODEL:-hunyuanvideo-community/HunyuanVideo}"
TRANSFORMER="${HUNYUAN_FP8_TRANSFORMER:-lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer}"

export WORLDODYSSEY_SGLANG_OFFLOAD_PRESET="${WORLDODYSSEY_SGLANG_OFFLOAD_PRESET:-memory}"
export WORLDODYSSEY_SGLANG_LOG_LEVEL="${WORLDODYSSEY_SGLANG_LOG_LEVEL:-debug}"
export WORLDODYSSEY_SGLANG_NUM_GPUS="${WORLDODYSSEY_SGLANG_NUM_GPUS:-1}"
export WORLDODYSSEY_SGLANG_TP_SIZE="${WORLDODYSSEY_SGLANG_TP_SIZE:-1}"
export WORLDODYSSEY_SGLANG_SP_DEGREE="${WORLDODYSSEY_SGLANG_SP_DEGREE:-$WORLDODYSSEY_SGLANG_NUM_GPUS}"

exec bash "$ROOT_DIR/scripts/serve_sglang_diffusion.sh" "$MODEL" \
    --transformer-path "$TRANSFORMER" \
    "$@"
