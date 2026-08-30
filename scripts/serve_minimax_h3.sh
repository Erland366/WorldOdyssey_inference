#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=../runtime-versions.env
source "$ROOT_DIR/runtime-versions.env"
# shellcheck source=minimax_h3_env.sh
source "$ROOT_DIR/scripts/minimax_h3_env.sh"

VARIANT="${1:-}"
if [[ "$VARIANT" != "fl2va" && "$VARIANT" != "ref2va" ]]; then
    echo "Usage: $0 {fl2va|ref2va} [additional sglang serve arguments]" >&2
    exit 2
fi
shift

SKIP_CUDA_PREFLIGHT=0
for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        SKIP_CUDA_PREFLIGHT=1
    fi
done

MODEL_PATH="${MINIMAX_H3_MODEL:-MiniMaxAI/MiniMax-H3}"
VENV_PATH="${WORLDODYSSEY_MINIMAX_H3_VENV:-$ROOT_DIR/.venv_sglang_h3}"
HOST="${WORLDODYSSEY_MINIMAX_H3_HOST:-127.0.0.1}"
NUM_GPUS="${WORLDODYSSEY_MINIMAX_H3_NUM_GPUS:-1}"
TP_SIZE="${WORLDODYSSEY_MINIMAX_H3_TP_SIZE:-1}"
ULYSSES_DEGREE="${WORLDODYSSEY_MINIMAX_H3_ULYSSES_DEGREE:-1}"
PERFORMANCE_MODE="${WORLDODYSSEY_MINIMAX_H3_PERFORMANCE_MODE:-memory}"

if [[ "$VARIANT" == "fl2va" ]]; then
    PORT="${WORLDODYSSEY_MINIMAX_H3_FL2VA_PORT:-30010}"
else
    PORT="${WORLDODYSSEY_MINIMAX_H3_REF2VA_PORT:-30011}"
fi

if [[ ! -x "$VENV_PATH/bin/sglang" ]]; then
    echo "MiniMax-H3 runtime not found at $VENV_PATH. Run scripts/install_minimax_h3.sh first." >&2
    exit 1
fi

# SGLang's H3 stack loads CUDA libraries from Python wheels and invokes Ninja
# by executable name. Prepare those library paths before activating the
# isolated environment.
worldodyssey_configure_minimax_h3_env "$VENV_PATH" "$ROOT_DIR"
source "$VENV_PATH/bin/activate"

for executable in ffmpeg ffprobe; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "MiniMax-H3 requires $executable in $VENV_PATH/bin. Run scripts/install_minimax_h3.sh again." >&2
        exit 1
    fi
done

python - "$WORLDODYSSEY_MINIMAX_H3_SGLANG_VERSION" <<'PY'
from importlib import metadata
import sys

installed = metadata.version("sglang")
if installed != sys.argv[1]:
    raise SystemExit(f"SGLang version mismatch: expected {sys.argv[1]}, got {installed}")
PY

if [[ "$SKIP_CUDA_PREFLIGHT" -eq 0 ]]; then
    python - <<'PY'
import torch

print(f"Torch CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit(
        "MiniMax-H3 CUDA preflight failed: torch.cuda.is_available() is false. "
        "The CUDA-12.9 H3 runtime requires an NVIDIA R525+ CUDA-12 driver."
    )
PY
fi

MEMORY_ARGS=()
if [[ "$PERFORMANCE_MODE" == "memory" ]]; then
    # Official lossless single-GPU recipe: stream the DiT and text encoder,
    # while leaving the repeatedly tiled VAE resident.
    MEMORY_ARGS=(
        --layerwise-offload-components dit,text_encoder
        --dit-offload-prefetch-size 1
        --dit-layerwise-resident-layers 0
        --enable-torch-compile false
    )
fi

ARGS=(
    serve
    --model-path "$MODEL_PATH"
    --model-variant "$VARIANT"
    --num-gpus "$NUM_GPUS"
    --tp-size "$TP_SIZE"
    --ulysses-degree "$ULYSSES_DEGREE"
    --encoder-parallel auto
    --performance-mode "$PERFORMANCE_MODE"
    --host "$HOST"
    --port "$PORT"
    --output-path "artifacts/minimax-h3-$VARIANT"
)
ARGS+=("${MEMORY_ARGS[@]}")
ARGS+=("$@")

cat <<EOF
Starting MiniMax-H3 $VARIANT server.

Backend route:
  export WORLDODYSSEY_MINIMAX_H3_${VARIANT^^}_BASE_URL=http://${HOST}:${PORT}

Runtime:
  venv=$VENV_PATH
  hf_home=$HF_HOME
  hf_hub_cache=$HUGGINGFACE_HUB_CACHE
  hf_hub_disable_xet=$HF_HUB_DISABLE_XET
  model=$MODEL_PATH
  variant=$VARIANT
  GPUs=$NUM_GPUS, TP=$TP_SIZE, Ulysses=$ULYSSES_DEGREE
  performance_mode=$PERFORMANCE_MODE

EOF

printf 'Command:\n  %q' "sglang"
printf ' %q' "${ARGS[@]}"
printf '\n'

exec sglang "${ARGS[@]}"
