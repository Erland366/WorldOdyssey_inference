#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${FASTWAN_T2V_MODEL:-FastVideo/FastWan2.1-T2V-1.3B-Diffusers}"
ATTENTION_BACKEND="${FASTWAN_ATTENTION_BACKEND:-video_sparse_attn}"
VSA_SPARSITY="${FASTWAN_VSA_SPARSITY:-0.5}"

export WORLDODYSSEY_SGLANG_NUM_GPUS="${WORLDODYSSEY_SGLANG_NUM_GPUS:-1}"

exec bash "$ROOT_DIR/scripts/serve_sglang_diffusion.sh" "$MODEL" \
    --attention-backend "$ATTENTION_BACKEND" \
    --VSA-sparsity "$VSA_SPARSITY" \
    "$@"
