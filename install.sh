#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INSTALL_SGLANG=1
INSTALL_MINIMAX_H3=0

usage() {
    cat <<'EOF'
Install WorldOdyssey Inference.

Usage:
  ./install.sh [options]

Options:
  --main-only       Install only the backend .venv (no GPU runtime).
  --skip-sglang     Do not create/update the isolated .venv_sglang runtime.
  --with-minimax-h3 Also install the isolated .venv_sglang_h3 runtime.
  -h, --help        Show this help message.

The default installation supports FastWan and Cosmos 3 through one pinned
SGLang Diffusion runtime. Direct FastVideo and direct Diffusers execution are
deprecated and are not installed.
EOF
}

while (($#)); do
    case "$1" in
        --main-only)
            INSTALL_SGLANG=0
            ;;
        --skip-sglang)
            INSTALL_SGLANG=0
            ;;
        --with-minimax-h3)
            INSTALL_MINIMAX_H3=1
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

section "Installing the main environment (.venv)"
# GPU runtimes live in isolated environments below, so the backend environment
# can exactly match uv.lock without preserving ABI-conflicting ML packages.
uv sync

if ((INSTALL_SGLANG)); then
    section "Installing the isolated SGLang Diffusion runtime (.venv_sglang)"
    bash scripts/install_sglang_diffusion.sh
fi

if ((INSTALL_MINIMAX_H3)); then
    section "Installing the isolated MiniMax-H3 runtime (.venv_sglang_h3)"
    bash scripts/install_minimax_h3.sh
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
  bash scripts/run_fastwan_t2v.sh

Then, in another shell, start the API backend:
  bash scripts/run_backend.sh
EOF
fi

if ((INSTALL_MINIMAX_H3)); then
    cat <<'EOF'

Start MiniMax-H3 first-frame image-to-video-and-audio:
  bash scripts/run_minimax_h3_fl2va.sh

For semantic reference conditioning, use:
  bash scripts/run_minimax_h3_ref2va.sh
EOF
fi

cat <<'EOF'

To serve Cosmos 3 through the same API, restart the SGLang model server with:
  bash scripts/run_cosmos3.sh

Then submit a Cosmos job through the unified backend:
  bash scripts/generate_cosmos3.sh
EOF
