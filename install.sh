#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INSTALL_SUBMODULE=1
INSTALL_SGLANG=1

usage() {
    cat <<'EOF'
Install WorldOdyssey Inference.

Usage:
  ./install.sh [options]

Options:
  --main-only       Install only the backend .venv (no submodule or GPU runtimes).
  --skip-submodule  Do not initialize the WorldOdyssey input submodule.
  --skip-sglang     Do not create/update the isolated .venv_sglang runtime.
  -h, --help        Show this help message.

The default installation supports FastWan and Cosmos 3 through one pinned
SGLang Diffusion runtime. Direct FastVideo and direct Diffusers execution are
deprecated and are not installed.
EOF
}

while (($#)); do
    case "$1" in
        --main-only)
            INSTALL_SUBMODULE=0
            INSTALL_SGLANG=0
            ;;
        --skip-submodule)
            INSTALL_SUBMODULE=0
            ;;
        --skip-sglang)
            INSTALL_SGLANG=0
            ;;
        --skip-cosmos)
            echo "Warning: --skip-cosmos is deprecated; Cosmos 3 is part of the unified SGLang runtime." >&2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: '$1' is required but was not found in PATH." >&2
        exit 1
    fi
}

section() {
    printf '\n==> %s\n' "$1"
}

require_command uv
if ((INSTALL_SUBMODULE)); then
    require_command git
fi

section "Installing the main environment (.venv)"
# GPU runtimes live in isolated environments below, so the backend environment
# can exactly match uv.lock without preserving ABI-conflicting ML packages.
uv sync

if ((INSTALL_SUBMODULE)); then
    section "Initializing the WorldOdyssey input submodule"
    git submodule update --init --recursive submodule/worldodyssey
fi

if ((INSTALL_SGLANG)); then
    section "Installing the isolated SGLang Diffusion runtime (.venv_sglang)"
    bash scripts/install_sglang_diffusion.sh
fi

section "Verifying the installation"
.venv/bin/python - <<'PY'
from importlib import metadata

for package in ("fastapi", "uvicorn", "pydantic"):
    print(f"  {package}=={metadata.version(package)}")
PY

cat <<EOF

Installation complete.

Activate the main environment:
  source .venv/bin/activate
EOF

if ((INSTALL_SGLANG)); then
    cat <<'EOF'

Start SGLang (GPU server):
  WORLDODYSSEY_SGLANG_NUM_GPUS=1 \
  bash scripts/serve_sglang_diffusion.sh FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
    --attention-backend video_sparse_attn \
    --VSA-sparsity 0.5

Then, in another shell, start the API backend:
  export WORLDODYSSEY_SGLANG_BASE_URL=http://127.0.0.1:30000
  source .venv/bin/activate
  python scripts/serve_video_backend.py --host 127.0.0.1 --port 8000
EOF
fi

cat <<'EOF'

To serve Cosmos 3 through the same API, restart the SGLang model server with:
  SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
  bash scripts/serve_sglang_diffusion.sh nvidia/Cosmos3-Nano

Then submit a Cosmos job through the unified backend:
  bash scripts/run_cosmos3.sh
EOF
