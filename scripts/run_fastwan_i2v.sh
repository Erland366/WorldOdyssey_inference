#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${FASTWAN_I2V_MODEL:-FastVideo/FastWan2.2-TI2V-5B-Diffusers}"

export WORLDODYSSEY_SGLANG_WORKLOAD_TYPE="${WORLDODYSSEY_SGLANG_WORKLOAD_TYPE:-i2v}"
export WORLDODYSSEY_SGLANG_NUM_GPUS="${WORLDODYSSEY_SGLANG_NUM_GPUS:-1}"

exec bash "$ROOT_DIR/scripts/serve_sglang_diffusion.sh" "$MODEL" "$@"
