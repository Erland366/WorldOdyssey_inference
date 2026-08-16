#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "scripts/install_cosmos3.sh is deprecated; Cosmos 3 now uses the unified SGLang runtime." >&2
exec bash "$ROOT_DIR/scripts/install_sglang_diffusion.sh" "$@"
