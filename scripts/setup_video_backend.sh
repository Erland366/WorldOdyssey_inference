#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required for setup." >&2
    exit 1
fi

git submodule update --init --recursive submodule/worldodyssey

# Keep already-installed local ML packages such as FastVideo intact. This setup
# only needs to ensure the server dependencies from the lockfile are present.
uv sync --inexact

# shellcheck source=/dev/null
source .venv/bin/activate

bash scripts/install_sglang_diffusion.sh

python - <<'PY'
from importlib import metadata

for package in ("fastapi", "uvicorn", "pydantic"):
    print(package, metadata.version(package))
PY

cat <<'EOF'

Video backend setup complete.

Start native SGLang first, for example:
  WORLDODYSSEY_SGLANG_NUM_GPUS=1 \
  bash scripts/serve_sglang_diffusion.sh FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
    --attention-backend video_sparse_attn \
    --VSA-sparsity 0.5

Then start the provider-neutral backend in another shell:
  export WORLDODYSSEY_SGLANG_BASE_URL=http://127.0.0.1:30000
  source .venv/bin/activate
  python scripts/serve_video_backend.py --host 127.0.0.1 --port 8000

Submit a local SGLang FastWan VSA job with:
  curl -X POST http://127.0.0.1:8000/v1/video/generations \
    -H "Content-Type: application/json" \
    -d '{"provider":"sglang","model":"FastVideo/FastWan2.1-T2V-1.3B-Diffusers","mode":"text_to_video","prompt":"A calm ocean wave at sunrise","options":{"height":448,"width":832,"num_frames":61,"timeout_seconds":600}}'
EOF
