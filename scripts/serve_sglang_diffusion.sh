#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${1:-${WORLDODYSSEY_SGLANG_MODEL:-FastVideo/FastWan2.1-T2V-1.3B-Diffusers}}"
if [[ $# -gt 0 ]]; then
  shift
fi

HOST="${WORLDODYSSEY_SGLANG_HOST:-127.0.0.1}"
PORT="${WORLDODYSSEY_SGLANG_PORT:-30000}"
NUM_GPUS="${WORLDODYSSEY_SGLANG_NUM_GPUS:-1}"
WORKLOAD_TYPE="${WORLDODYSSEY_SGLANG_WORKLOAD_TYPE:-t2v}"
LOG_LEVEL="${WORLDODYSSEY_SGLANG_LOG_LEVEL:-info}"

if [[ ! -x ".venv_sglangcuda12/bin/sglang" ]]; then
  echo "SGLang runtime not found at .venv_sglangcuda12/bin/sglang. Run scripts/install_sglang_diffusion.sh first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source .venv_sglangcuda12/bin/activate

export PATH="$ROOT_DIR/.venv_sglangcuda12/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME="$ROOT_DIR/.venv_sglangcuda12/lib/python3.12/site-packages/nvidia"
export PYTHONUNBUFFERED=1

cat <<EOF
Starting native SGLang diffusion server.

Backend environment for this server:
  export WORLDODYSSEY_SGLANG_BASE_URL=http://${HOST}:${PORT}

Optional backend model hint:
  export WORLDODYSSEY_SGLANG_MODEL=${MODEL_PATH}

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
