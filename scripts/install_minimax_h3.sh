#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=../runtime-versions.env
source "$ROOT_DIR/runtime-versions.env"
# shellcheck source=minimax_h3_env.sh
source "$ROOT_DIR/scripts/minimax_h3_env.sh"

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
    "ninja==$WORLDODYSSEY_MINIMAX_H3_NINJA_VERSION" \
    "static-ffmpeg==$WORLDODYSSEY_MINIMAX_H3_STATIC_FFMPEG_VERSION"

# SGLang's default wheel stack is CUDA 13, but the documented CUDA-12 path is
# a post-install override. Keep this sequence explicit so R570 CUDA-12 drivers
# can run H3 without solving an impossible mixed dependency graph.
uv pip install \
    --python "$VENV_PATH/bin/python" \
    --force-reinstall \
    "torch==$WORLDODYSSEY_MINIMAX_H3_TORCH_VERSION" \
    "torchaudio==$WORLDODYSSEY_MINIMAX_H3_TORCHAUDIO_VERSION" \
    "torchvision==$WORLDODYSSEY_MINIMAX_H3_TORCHVISION_VERSION" \
    --index-url "https://download.pytorch.org/whl/$WORLDODYSSEY_MINIMAX_H3_TORCH_BACKEND"

uv pip install \
    --python "$VENV_PATH/bin/python" \
    --force-reinstall \
    "sglang-kernel==$WORLDODYSSEY_MINIMAX_H3_SGLANG_KERNEL_VERSION" \
    --index-url "https://docs.sglang.ai/whl/$WORLDODYSSEY_MINIMAX_H3_TORCH_BACKEND/"

uv pip install \
    --python "$VENV_PATH/bin/python" \
    --force-reinstall \
    "sgl-deep-gemm==$WORLDODYSSEY_MINIMAX_H3_DEEP_GEMM_VERSION" \
    --index-url "https://docs.sglang.ai/whl/$WORLDODYSSEY_MINIMAX_H3_TORCH_BACKEND/" \
    --no-deps

worldodyssey_configure_minimax_h3_env "$VENV_PATH" "$ROOT_DIR"
source "$VENV_PATH/bin/activate"

# H3 emits joint audio/video output and validates it with ffprobe after the
# expensive generation pass. Fetch both static tools during installation and
# expose their canonical names inside the isolated venv.
python - "$VENV_PATH" <<'PY'
from pathlib import Path
import sys

from static_ffmpeg import run

venv_bin = Path(sys.argv[1]).resolve() / "bin"
ffmpeg, ffprobe = run.get_or_fetch_platform_executables_else_raise()
for name, target in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
    link = venv_bin / name
    link.unlink(missing_ok=True)
    link.symlink_to(Path(target).resolve())
    print(f"{name} executable: {link} -> {link.resolve()}")
PY

python - \
    "$WORLDODYSSEY_MINIMAX_H3_SGLANG_VERSION" \
    "$WORLDODYSSEY_MINIMAX_H3_TORCH_VERSION" \
    "$WORLDODYSSEY_MINIMAX_H3_TRANSFORMERS_VERSION" \
    "$WORLDODYSSEY_MINIMAX_H3_NINJA_VERSION" \
    "$WORLDODYSSEY_MINIMAX_H3_TORCHAUDIO_VERSION" \
    "$WORLDODYSSEY_MINIMAX_H3_TORCHVISION_VERSION" \
    "$WORLDODYSSEY_MINIMAX_H3_STATIC_FFMPEG_VERSION" \
    "$WORLDODYSSEY_MINIMAX_H3_SGLANG_KERNEL_VERSION" \
    "$WORLDODYSSEY_MINIMAX_H3_DEEP_GEMM_VERSION" <<'PY'
from importlib import metadata
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

expected = {
    "sglang": sys.argv[1],
    "torch": sys.argv[2],
    "transformers": sys.argv[3],
    "ninja": sys.argv[4],
    "torchaudio": sys.argv[5],
    "torchvision": sys.argv[6],
    "static-ffmpeg": sys.argv[7],
    "sglang-kernel": sys.argv[8],
    "sgl-deep-gemm": sys.argv[9],
}
for package, version in expected.items():
    installed = metadata.version(package)
    if installed != version and not installed.startswith(f"{version}+"):
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

for executable in ("ffmpeg", "ffprobe"):
    executable_path = shutil.which(executable)
    if executable_path is None:
        raise SystemExit(f"{executable} is missing from the MiniMax-H3 runtime PATH")
    subprocess.run(
        [executable_path, "-version"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"{executable} executable: {executable_path}")

import torch

print(f"Torch CUDA runtime: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit(
        "MiniMax-H3 CUDA preflight failed: torch.cuda.is_available() is false. "
        "The CUDA-12.9 H3 runtime requires an NVIDIA R525+ CUDA-12 driver."
    )

import torchaudio  # noqa: F401
import torchvision  # noqa: F401
import sglang.multimodal_gen.runtime.server_args  # noqa: F401
print("SGLang Diffusion runtime import: ok")
PY

MINIMAX_H3_SOURCE="$(python - <<'PY'
from pathlib import Path
import sysconfig

print(
    Path(sysconfig.get_paths()["purelib"])
    / "sglang/multimodal_gen/runtime/models/dits/minimax_h3.py"
)
PY
)"
python "$ROOT_DIR/scripts/patch_minimax_h3_adaln_cache.py" "$MINIMAX_H3_SOURCE"

cat <<EOF

MiniMax-H3 SGLang runtime is ready at:
  $VENV_PATH

Start the primary text/one-image server:
  bash scripts/run_minimax_h3_fl2va.sh

Start semantic reference conditioning separately:
  bash scripts/run_minimax_h3_ref2va.sh
EOF
