#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PROFILE="${SGLANG_DIFFUSION_PROFILE:-unified}"
PYTHON_SPEC="${SGLANG_DIFFUSION_PYTHON:-3.12}"

case "$PROFILE" in
    unified | hunyuan-fp8)
        DEFAULT_VENV_PATH="$ROOT_DIR/.venv_sglang"
        ;;
    fastwan-vsa | fastwan-vsa-legacy)
        DEFAULT_VENV_PATH="$ROOT_DIR/.venv_sglangcuda12"
        ;;
    *)
        echo "Unsupported SGLANG_DIFFUSION_PROFILE=$PROFILE. Expected unified, hunyuan-fp8, or fastwan-vsa-legacy." >&2
        exit 1
        ;;
esac

VENV_PATH="${SGLANG_DIFFUSION_VENV:-$DEFAULT_VENV_PATH}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required for environment creation and dependency installation." >&2
    exit 1
fi

if [[ ! -d "$VENV_PATH" ]]; then
    uv venv -p "$PYTHON_SPEC" "$VENV_PATH"
fi

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

case "$PROFILE" in
    fastwan-vsa | fastwan-vsa-legacy)
        uv pip install --prerelease=allow \
            "sglang[diffusion]==0.5.5" \
            "cuda-python==12.9.6" \
            "pytest==9.0.3"
        ;;
    unified | hunyuan-fp8)
        uv pip install --torch-backend cu128 --prerelease=allow \
            "torch==2.9.1" \
            "torchvision==0.24.1" \
            "sglang[diffusion]==0.5.10.post1" \
            "tokenizers==0.22.1" \
            "nvidia-modelopt==0.44.0" \
            "accelerate==1.13.0" \
            "diffusers==0.37.0" \
            "transformers==5.3.0" \
            "cuda-python==12.9.0" \
            "pytest==9.0.3"
        python scripts/patch_sglang_hunyuan_fp8.py --venv "$VENV_PATH"
        ;;
esac

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

python - "$PROFILE" <<'PY'
from importlib import metadata
import sys

profile = sys.argv[1]
if profile in {"fastwan-vsa", "fastwan-vsa-legacy"}:
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
    unexpected_packages = (
        "nvidia-cuda-runtime-cu13",
        "nvidia-cuda-nvrtc-cu13",
        "nvidia-nccl-cu13",
        "sglang-kernel",
    )
else:
    packages = (
        "torch",
        "torchvision",
        "sglang",
        "sglang-kernel",
        "flashinfer-python",
        "diffusers",
        "transformers",
        "tokenizers",
        "nvidia-modelopt",
        "accelerate",
        "cuda-python",
        "cuda-bindings",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-runtime-cu13",
    )
    unexpected_packages = (
        "nvidia-cuda-runtime-cu13",
        "nvidia-cuda-nvrtc-cu13",
        "nvidia-nccl-cu13",
    )

for name in packages:
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name} not installed")

installed_unexpected = []
for name in unexpected_packages:
    try:
        installed_unexpected.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        continue

if installed_unexpected:
    raise SystemExit(
        "Unexpected packages installed for " + profile + ": " + ", ".join(installed_unexpected)
    )
PY

cat <<EOF

SGLang Diffusion profile installed:
  $PROFILE

Environment:
  $VENV_PATH

Use these runtime guards before running SGLang Diffusion on this host:
  source "$VENV_PATH/bin/activate"
  export PATH="$VENV_PATH/bin:/usr/local/bin:/usr/bin:/bin"
  export CC=/usr/bin/gcc
  export CXX=/usr/bin/g++
  export CUDA_HOME="$CUDA_HOME_PATH"

See references/sglang-diffusion.md for FastWan VSA commands and docs/video-backend-runbook.md for FP8 commands.
EOF
