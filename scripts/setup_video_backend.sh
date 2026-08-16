#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Note: scripts/setup_video_backend.sh is kept for compatibility; use ./install.sh for new setups."
exec "$ROOT_DIR/install.sh" "$@"
