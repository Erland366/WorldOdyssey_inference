#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=../runtime-versions.env
source "$ROOT_DIR/runtime-versions.env"

MODEL_PATH="${1:-${WORLDODYSSEY_SGLANG_MODEL:-FastVideo/FastWan2.1-T2V-1.3B-Diffusers}}"
if [[ $# -gt 0 ]]; then
  shift
fi

# SGLang moved the standalone --VSA-sparsity flag into the structured
# --attention-backend-config option. Keep the repository's established command
# compatible while always invoking the pinned source with its native shape.
EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --VSA-sparsity)
      if [[ $# -lt 2 ]]; then
        echo "--VSA-sparsity requires a numeric value." >&2
        exit 2
      fi
      EXTRA_ARGS+=(--attention-backend-config "{\"VSA_sparsity\":$2}")
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${EXTRA_ARGS[@]}"

HOST="${WORLDODYSSEY_SGLANG_HOST:-127.0.0.1}"
PORT="${WORLDODYSSEY_SGLANG_PORT:-30000}"
NUM_GPUS="${WORLDODYSSEY_SGLANG_NUM_GPUS:-1}"
WORKLOAD_TYPE="${WORLDODYSSEY_SGLANG_WORKLOAD_TYPE:-t2v}"
LOG_LEVEL="${WORLDODYSSEY_SGLANG_LOG_LEVEL:-info}"
ENTRYPOINT="${WORLDODYSSEY_SGLANG_ENTRYPOINT:-native}"
VENV_PATH="${WORLDODYSSEY_SGLANG_VENV:-$ROOT_DIR/.venv_sglang}"
SGLANG_SOURCE_PATH="${WORLDODYSSEY_SGLANG_SOURCE:-$ROOT_DIR/.deps/sglang}"
SGLANG_SOURCE_REV="${WORLDODYSSEY_SGLANG_SOURCE_REV_OVERRIDE:-$WORLDODYSSEY_SGLANG_SOURCE_REV}"
VIDEO_API_FORMAT="${WORLDODYSSEY_SGLANG_VIDEO_API_FORMAT:-multipart}"
BACKEND="${WORLDODYSSEY_SGLANG_BACKEND:-sglang}"
TP_SIZE="${WORLDODYSSEY_SGLANG_TP_SIZE:-1}"
SP_DEGREE="${WORLDODYSSEY_SGLANG_SP_DEGREE:-1}"
OUTPUT_PATH="${WORLDODYSSEY_SGLANG_OUTPUT_PATH:-artifacts/video-backend}"
OFFLOAD_PRESET="${WORLDODYSSEY_SGLANG_OFFLOAD_PRESET:-none}"
VAE_TILE_SAMPLE_MIN_HEIGHT="${WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_MIN_HEIGHT:-128}"
VAE_TILE_SAMPLE_MIN_WIDTH="${WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_MIN_WIDTH:-128}"
VAE_TILE_SAMPLE_STRIDE_HEIGHT="${WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_STRIDE_HEIGHT:-96}"
VAE_TILE_SAMPLE_STRIDE_WIDTH="${WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_STRIDE_WIDTH:-96}"
VAE_TILE_SAMPLE_MIN_NUM_FRAMES="${WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_MIN_NUM_FRAMES:-8}"
VAE_TILE_SAMPLE_STRIDE_NUM_FRAMES="${WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_STRIDE_NUM_FRAMES:-4}"

case "$ENTRYPOINT" in
  legacy-wrapper | native)
    ;;
  *)
    echo "Unsupported WORLDODYSSEY_SGLANG_ENTRYPOINT=$ENTRYPOINT. Expected legacy-wrapper or native." >&2
    exit 1
    ;;
esac

OFFLOAD_ARGS=()
case "$OFFLOAD_PRESET" in
  none)
    ;;
  memory)
    OFFLOAD_ARGS=(
      --dit-layerwise-offload
      --dit-cpu-offload false
      --dit-offload-prefetch-size 0
      --text-encoder-cpu-offload
      --image-encoder-cpu-offload
      --vae-cpu-offload
      --pin-cpu-memory
      --vae-tiling
      --vae-slicing
      --vae-config.tile-sample-min-height "$VAE_TILE_SAMPLE_MIN_HEIGHT"
      --vae-config.tile-sample-min-width "$VAE_TILE_SAMPLE_MIN_WIDTH"
      --vae-config.tile-sample-stride-height "$VAE_TILE_SAMPLE_STRIDE_HEIGHT"
      --vae-config.tile-sample-stride-width "$VAE_TILE_SAMPLE_STRIDE_WIDTH"
      --vae-config.tile-sample-min-num-frames "$VAE_TILE_SAMPLE_MIN_NUM_FRAMES"
      --vae-config.tile-sample-stride-num-frames "$VAE_TILE_SAMPLE_STRIDE_NUM_FRAMES"
    )
    ;;
  *)
    echo "Unsupported WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=$OFFLOAD_PRESET. Expected none or memory." >&2
    exit 1
    ;;
esac

if [[ "$ENTRYPOINT" != "native" && "$OFFLOAD_PRESET" != "none" ]]; then
  echo "WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=$OFFLOAD_PRESET is only supported with the native SGLang entrypoint." >&2
  exit 1
fi

if [[ "$OFFLOAD_PRESET" == "memory" && -z "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE+x}" ]]; then
  export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1
fi
if [[ "$OFFLOAD_PRESET" == "memory" && -z "${USE_TRITON_W8A8_FP8_KERNEL+x}" ]]; then
  export USE_TRITON_W8A8_FP8_KERNEL=1
fi
if [[ "$OFFLOAD_PRESET" == "memory" && -z "${SGLANG_DISABLE_FLASHINFER_ROPE+x}" ]]; then
  export SGLANG_DISABLE_FLASHINFER_ROPE=1
fi
if [[ "$OFFLOAD_PRESET" == "memory" && -z "${PYTORCH_CUDA_ALLOC_CONF+x}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

case "${SGLANG_CACHE_DIT_ENABLED:-false}" in
  true | True | TRUE | 1 | yes | Yes | YES)
    if [[ "$OFFLOAD_PRESET" == "memory" ]]; then
      echo "WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory cannot be combined with SGLANG_CACHE_DIT_ENABLED=${SGLANG_CACHE_DIT_ENABLED}." >&2
      exit 1
    fi
    ;;
esac

if [[ ! -x "$VENV_PATH/bin/sglang" ]]; then
  echo "SGLang runtime not found at $VENV_PATH/bin/sglang. Run scripts/install_sglang_diffusion.sh first." >&2
  exit 1
fi
if [[ ! -f "$SGLANG_SOURCE_PATH/python/sglang/__init__.py" ]]; then
  echo "Pinned SGLang source not found at $SGLANG_SOURCE_PATH. Run scripts/install_sglang_diffusion.sh first." >&2
  exit 1
fi
INSTALLED_SOURCE_REV="$(git -C "$SGLANG_SOURCE_PATH" rev-parse HEAD 2>/dev/null || true)"
if [[ "$INSTALLED_SOURCE_REV" != "$SGLANG_SOURCE_REV" ]]; then
  echo "Pinned SGLang source mismatch: expected $SGLANG_SOURCE_REV, got ${INSTALLED_SOURCE_REV:-missing}." >&2
  echo "Run scripts/install_sglang_diffusion.sh to repair it." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

CUDA_HOME_PATH="${SGLANG_DIFFUSION_CUDA_HOME:-/usr/local/cuda}"
if [[ ! -x "$CUDA_HOME_PATH/bin/nvcc" || ! -e "$CUDA_HOME_PATH/lib64/libcudart.so.12" ]]; then
  echo "A complete CUDA 12 toolkit is required at $CUDA_HOME_PATH (missing nvcc or libcudart)." >&2
  echo "Set SGLANG_DIFFUSION_CUDA_HOME to a compatible toolkit root." >&2
  exit 1
fi
if ! "$CUDA_HOME_PATH/bin/nvcc" --version | grep -q 'release 12\.'; then
  echo "Cosmos/FastWan require a CUDA 12 nvcc; found:" >&2
  "$CUDA_HOME_PATH/bin/nvcc" --version >&2
  exit 1
fi

export PATH="$VENV_PATH/bin:$CUDA_HOME_PATH/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME="$CUDA_HOME_PATH"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SGLANG_SOURCE_PATH/python${PYTHONPATH:+:$PYTHONPATH}"

cat <<EOF
Starting native SGLang diffusion server.

Backend environment for this server:
  export WORLDODYSSEY_SGLANG_BASE_URL=http://${HOST}:${PORT}
  export WORLDODYSSEY_SGLANG_VIDEO_API_FORMAT=${VIDEO_API_FORMAT}

Optional backend model hint:
  export WORLDODYSSEY_SGLANG_MODEL=${MODEL_PATH}

Runtime:
  venv=${VENV_PATH}
  entrypoint=${ENTRYPOINT}
  offload_preset=${OFFLOAD_PRESET}
  SGLANG_ENABLE_DETERMINISTIC_INFERENCE=${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-unset}
  USE_TRITON_W8A8_FP8_KERNEL=${USE_TRITON_W8A8_FP8_KERNEL:-unset}
  SGLANG_DISABLE_FLASHINFER_ROPE=${SGLANG_DISABLE_FLASHINFER_ROPE:-unset}
  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-unset}
  vae_tile_sample_min_height=${VAE_TILE_SAMPLE_MIN_HEIGHT}
  vae_tile_sample_min_width=${VAE_TILE_SAMPLE_MIN_WIDTH}
  vae_tile_sample_stride_height=${VAE_TILE_SAMPLE_STRIDE_HEIGHT}
  vae_tile_sample_stride_width=${VAE_TILE_SAMPLE_STRIDE_WIDTH}
  vae_tile_sample_min_num_frames=${VAE_TILE_SAMPLE_MIN_NUM_FRAMES}
  vae_tile_sample_stride_num_frames=${VAE_TILE_SAMPLE_STRIDE_NUM_FRAMES}
  CUDA_HOME=${CUDA_HOME}
  SGLang_source=${SGLANG_SOURCE_PATH}

EOF

if [[ "$ENTRYPOINT" == "native" ]]; then
  NATIVE_ARGS=(
    serve
    --model-type diffusion
    --model-path "$MODEL_PATH"
    --backend "$BACKEND"
    --num-gpus "$NUM_GPUS"
    --tp-size "$TP_SIZE"
    --sp-degree "$SP_DEGREE"
    --host "$HOST"
    --port "$PORT"
    --output-path "$OUTPUT_PATH"
    --log-level "$LOG_LEVEL"
  )
  NATIVE_ARGS+=("${OFFLOAD_ARGS[@]}")
  NATIVE_ARGS+=("$@")

  printf 'Command:\n  sglang'
  printf ' %q' "${NATIVE_ARGS[@]}"
  printf '\n'

  exec sglang "${NATIVE_ARGS[@]}"
fi

cat <<EOF
Command:
  python scripts/sglang_diffusion_serve.py --model-path ${MODEL_PATH} --workload-type ${WORKLOAD_TYPE} --num-gpus ${NUM_GPUS} --host ${HOST} --port ${PORT} --log-level ${LOG_LEVEL} $*
EOF

exec python scripts/sglang_diffusion_serve.py \
  --model-path "$MODEL_PATH" \
  --workload-type "$WORKLOAD_TYPE" \
  --num-gpus "$NUM_GPUS" \
  --host "$HOST" \
  --port "$PORT" \
  --log-level "$LOG_LEVEL" \
  "$@"
