#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# TP4 needs online AdaLN on this memory-capped Slurm allocation. It leaves the
# 24.2 GiB AdaLN projection weights out of each worker's resident model state.
export WORLDODYSSEY_MINIMAX_H3_NUM_GPUS="${WORLDODYSSEY_MINIMAX_H3_NUM_GPUS:-4}"
export WORLDODYSSEY_MINIMAX_H3_TP_SIZE="${WORLDODYSSEY_MINIMAX_H3_TP_SIZE:-4}"

exec bash "$ROOT_DIR/scripts/serve_minimax_h3.sh" fl2va \
    --minimax-h3-adaln-online \
    --minimax-h3-adaln-plan-width 3 \
    "$@"
