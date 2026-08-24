#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=../runtime-versions.env
source "$ROOT_DIR/runtime-versions.env"

VENV_PATH="${WORLDODYSSEY_MINIMAX_H3_VENV:-$ROOT_DIR/.venv_sglang_h3}"
PYTHON_SPEC="${WORLDODYSSEY_MINIMAX_H3_PYTHON:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required for MiniMax-H3 environment creation." >&2
    exit 1
fi

if [[ ! -d "$VENV_PATH" ]]; then
    uv venv -p "$PYTHON_SPEC" "$VENV_PATH"
fi

uv pip install \
    --python "$VENV_PATH/bin/python" \
    --prerelease=allow \
    "sglang[diffusion]==$WORLDODYSSEY_MINIMAX_H3_SGLANG_VERSION" \
    "ninja==$WORLDODYSSEY_MINIMAX_H3_NINJA_VERSION"

"$VENV_PATH/bin/python" - \
    "$WORLDODYSSEY_MINIMAX_H3_SGLANG_VERSION" \
    "$WORLDODYSSEY_MINIMAX_H3_TORCH_VERSION" \
    "$WORLDODYSSEY_MINIMAX_H3_TRANSFORMERS_VERSION" \
    "$WORLDODYSSEY_MINIMAX_H3_NINJA_VERSION" <<'PY'
from importlib import metadata
from pathlib import Path
import sys
import sysconfig

expected = {
    "sglang": sys.argv[1],
    "torch": sys.argv[2],
    "transformers": sys.argv[3],
    "ninja": sys.argv[4],
}
for package, version in expected.items():
    installed = metadata.version(package)
    if installed != version:
        raise SystemExit(f"{package} version mismatch: expected {version}, got {installed}")
    print(f"{package}=={installed}")

module_path = (
    Path(sysconfig.get_paths()["purelib"])
    / "sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3"
)
if not module_path.is_dir():
    raise SystemExit(f"MiniMax-H3 runtime module is missing: {module_path}")
print(f"MiniMax-H3 runtime module: {module_path}")

ninja_path = Path(sys.executable).parent / "ninja"
if not ninja_path.is_file():
    raise SystemExit(f"Ninja executable is missing: {ninja_path}")
print(f"Ninja executable: {ninja_path}")
PY

cat <<EOF

MiniMax-H3 SGLang runtime is ready at:
  $VENV_PATH

Start the primary text/one-image server:
  bash scripts/run_minimax_h3_fl2va.sh

Start semantic reference conditioning separately:
  bash scripts/run_minimax_h3_ref2va.sh
EOF
