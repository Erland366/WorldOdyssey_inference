#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${SGLANG_DIFFUSION_VENV:-$ROOT_DIR/.venv_sglangcuda12}"
PYTHON_SPEC="${SGLANG_DIFFUSION_PYTHON:-3.12}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required for environment creation and dependency installation." >&2
    exit 1
fi

if [[ ! -d "$VENV_PATH" ]]; then
    uv venv -p "$PYTHON_SPEC" "$VENV_PATH"
fi

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

uv pip install --prerelease=allow \
    "sglang[diffusion]==0.5.5" \
    "cuda-python==12.9.6" \
    "pytest==9.0.3"

CUDA_HOME_PATH="$(
    python - <<'PY'
from pathlib import Path
import site

for site_dir in site.getsitepackages():
    candidate = Path(site_dir) / "nvidia"
    if (candidate / "cuda_runtime" / "lib" / "libcudart.so.12").exists():
        print(candidate)
        break
else:
    raise SystemExit("Could not locate the venv NVIDIA CUDA runtime package")
PY
)"

python - <<'PY'
from importlib import metadata

packages = (
    "torch",
    "torchvision",
    "torchaudio",
    "sglang",
    "sgl-kernel",
    "vsa",
    "diffusers",
    "triton",
    "cuda-python",
    "cuda-bindings",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-nccl-cu12",
)

for name in packages:
    print(f"{name}=={metadata.version(name)}")

cuda13_packages = (
    "nvidia-cuda-runtime-cu13",
    "nvidia-cuda-nvrtc-cu13",
    "nvidia-nccl-cu13",
    "sglang-kernel",
)
installed_cuda13 = []
for name in cuda13_packages:
    try:
        installed_cuda13.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass

if installed_cuda13:
    raise SystemExit(
        "Unexpected CUDA-13 SGLang packages installed: " + ", ".join(installed_cuda13)
    )
PY

cat <<EOF

SGLang Diffusion environment installed at:
  $VENV_PATH

Use these runtime guards before running SGLang Diffusion on this host:
  source "$VENV_PATH/bin/activate"
  export PATH="$VENV_PATH/bin:/usr/local/bin:/usr/bin:/bin"
  export CC=/usr/bin/gcc
  export CXX=/usr/bin/g++
  export CUDA_HOME="$CUDA_HOME_PATH"

See references/sglang-diffusion.md for validation and FastWan VSA run commands.
EOF
