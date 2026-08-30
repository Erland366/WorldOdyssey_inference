#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=../runtime-versions.env
source "$ROOT_DIR/runtime-versions.env"
# shellcheck source=minimax_h3_env.sh
source "$ROOT_DIR/scripts/minimax_h3_env.sh"

VARIANT="${1:-fl2va}"
shift || true

case "$VARIANT" in
    fl2va)
        MODEL_SUBDIR="FL2VA"
        ;;
    ref2va)
        MODEL_SUBDIR="Ref2VA"
        ;;
    *)
        echo "Usage: $0 {fl2va|ref2va} [additional hf download arguments]" >&2
        exit 2
        ;;
esac

MODEL_PATH="${MINIMAX_H3_MODEL:-MiniMaxAI/MiniMax-H3}"
VENV_PATH="${WORLDODYSSEY_MINIMAX_H3_VENV:-$ROOT_DIR/.venv_sglang_h3}"
MAX_WORKERS="${WORLDODYSSEY_HF_DOWNLOAD_MAX_WORKERS:-4}"

if [[ ! -x "$VENV_PATH/bin/hf" ]]; then
    echo "MiniMax-H3 runtime not found at $VENV_PATH. Run scripts/install_minimax_h3.sh first." >&2
    exit 1
fi

worldodyssey_configure_minimax_h3_env "$VENV_PATH" "$ROOT_DIR"

cat <<EOF
Downloading MiniMax-H3 $VARIANT model files.

Runtime:
  venv=$VENV_PATH
  hf_home=$HF_HOME
  hf_hub_cache=$HUGGINGFACE_HUB_CACHE
  hf_hub_disable_xet=$HF_HUB_DISABLE_XET
  model=$MODEL_PATH
  include=$MODEL_SUBDIR/**
  max_workers=$MAX_WORKERS

EOF

source "$VENV_PATH/bin/activate"
exec hf download "$MODEL_PATH" --include "$MODEL_SUBDIR/**" --max-workers "$MAX_WORKERS" "$@"
