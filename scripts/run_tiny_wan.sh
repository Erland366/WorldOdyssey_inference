#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${TINY_WAN_MODEL:-Erland/tiny-wan2.1-t2v-debug}"

export WORLDODYSSEY_SGLANG_NUM_GPUS="${WORLDODYSSEY_SGLANG_NUM_GPUS:-1}"

exec bash "$ROOT_DIR/scripts/serve_sglang_diffusion.sh" "$MODEL" "$@"
