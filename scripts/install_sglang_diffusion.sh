#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=../runtime-versions.env
source "$ROOT_DIR/runtime-versions.env"
PROFILE="${SGLANG_DIFFUSION_PROFILE:-unified}"
PYTHON_SPEC="${SGLANG_DIFFUSION_PYTHON:-3.12}"
TORCH_BACKEND="${SGLANG_DIFFUSION_TORCH_BACKEND:-cu128}"
SOURCE_PATH="${WORLDODYSSEY_SGLANG_SOURCE:-$ROOT_DIR/.deps/sglang}"
SOURCE_REV="${WORLDODYSSEY_SGLANG_SOURCE_REV_OVERRIDE:-$WORLDODYSSEY_SGLANG_SOURCE_REV}"
VSA_SOURCE_PATH="${WORLDODYSSEY_VSA_SOURCE:-$ROOT_DIR/.deps/fastvideo-vsa}"
VSA_SOURCE_REV="${WORLDODYSSEY_VSA_SOURCE_REV_OVERRIDE:-$WORLDODYSSEY_VSA_SOURCE_REV}"

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

if [[ "$PROFILE" == "unified" || "$PROFILE" == "hunyuan-fp8" ]]; then
    if ! command -v git >/dev/null 2>&1; then
        echo "git is required to install the pinned Cosmos-capable SGLang source." >&2
        exit 1
    fi
    if [[ -e "$SOURCE_PATH" && ! -d "$SOURCE_PATH/.git" ]]; then
        echo "SGLang source path exists but is not a Git checkout: $SOURCE_PATH" >&2
        exit 1
    fi
    if [[ ! -d "$SOURCE_PATH/.git" ]]; then
        mkdir -p "$(dirname "$SOURCE_PATH")"
        git clone --filter=blob:none --no-checkout https://github.com/sgl-project/sglang.git "$SOURCE_PATH"
    fi
    git -C "$SOURCE_PATH" fetch --depth 1 origin "$SOURCE_REV"
    git -C "$SOURCE_PATH" checkout --detach "$SOURCE_REV"
    INSTALLED_SOURCE_REV="$(git -C "$SOURCE_PATH" rev-parse HEAD)"
    if [[ "$INSTALLED_SOURCE_REV" != "$SOURCE_REV" ]]; then
        echo "SGLang source revision mismatch: expected $SOURCE_REV, got $INSTALLED_SOURCE_REV" >&2
        exit 1
    fi
    if [[ -e "$VSA_SOURCE_PATH" && ! -d "$VSA_SOURCE_PATH/.git" ]]; then
        echo "VSA source path exists but is not a Git checkout: $VSA_SOURCE_PATH" >&2
        exit 1
    fi
    if [[ ! -d "$VSA_SOURCE_PATH/.git" ]]; then
        mkdir -p "$(dirname "$VSA_SOURCE_PATH")"
        git clone --filter=blob:none --no-checkout https://github.com/hao-ai-lab/FastVideo.git "$VSA_SOURCE_PATH"
    fi
    git -C "$VSA_SOURCE_PATH" fetch --depth 1 origin "$VSA_SOURCE_REV"
    git -C "$VSA_SOURCE_PATH" checkout --detach "$VSA_SOURCE_REV"
    git -C "$VSA_SOURCE_PATH" submodule update --init --recursive csrc/attn/video_sparse_attn/tk
    INSTALLED_VSA_SOURCE_REV="$(git -C "$VSA_SOURCE_PATH" rev-parse HEAD)"
    if [[ "$INSTALLED_VSA_SOURCE_REV" != "$VSA_SOURCE_REV" ]]; then
        echo "VSA source revision mismatch: expected $VSA_SOURCE_REV, got $INSTALLED_VSA_SOURCE_REV" >&2
        exit 1
    fi
fi

if [[ ! -d "$VENV_PATH" ]]; then
    uv venv -p "$PYTHON_SPEC" "$VENV_PATH"
fi

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

CUDA_HOME_PATH="${SGLANG_DIFFUSION_CUDA_HOME:-/usr/local/cuda}"
if [[ ! -x "$CUDA_HOME_PATH/bin/nvcc" || ! -e "$CUDA_HOME_PATH/lib64/libcudart.so.12" ]]; then
    echo "A complete CUDA 12 toolkit (nvcc and libcudart) is required at $CUDA_HOME_PATH." >&2
    echo "Set SGLANG_DIFFUSION_CUDA_HOME to a compatible toolkit root." >&2
    exit 1
fi
if ! "$CUDA_HOME_PATH/bin/nvcc" --version | grep -q 'release 12\.'; then
    echo "Cosmos/FastWan require a CUDA 12 nvcc; found:" >&2
    "$CUDA_HOME_PATH/bin/nvcc" --version >&2
    exit 1
fi

case "$PROFILE" in
    fastwan-vsa | fastwan-vsa-legacy)
        uv pip install --prerelease=allow \
            "sglang[diffusion]==0.5.5" \
            "cuda-python==12.9.6" \
            "pytest==9.0.3"
        ;;
    unified | hunyuan-fp8)
        # Install the last validated stable wheel first to provide the CLI and
        # compiled support packages without pulling the current CUDA-13 stack.
        SGLANG_BASE_MARKER="$VENV_PATH/.worldodyssey-sglang-base-version"
        if [[ ! -f "$SGLANG_BASE_MARKER" ]] && python - "$WORLDODYSSEY_SGLANG_VERSION" <<'PY' >/dev/null 2>&1
from importlib import metadata
import sys

assert metadata.version("sglang") == sys.argv[1]
for package in ("diffusers", "nvidia-modelopt", "accelerate", "cuda-python"):
    metadata.version(package)
PY
        then
            printf '%s\n' "$WORLDODYSSEY_SGLANG_VERSION" >"$SGLANG_BASE_MARKER"
        fi
        if [[ ! -f "$SGLANG_BASE_MARKER" || "$(<"$SGLANG_BASE_MARKER")" != "$WORLDODYSSEY_SGLANG_VERSION" ]]; then
            uv pip install --torch-backend "$TORCH_BACKEND" --prerelease=allow \
                "sglang[diffusion]==$WORLDODYSSEY_SGLANG_VERSION" \
                "tokenizers==0.22.1" \
                "nvidia-modelopt==0.44.0" \
                "accelerate==1.13.0" \
                "diffusers==0.37.0" \
                "cuda-python==12.9.0" \
                "pytest==9.0.3"
            printf '%s\n' "$WORLDODYSSEY_SGLANG_VERSION" >"$SGLANG_BASE_MARKER"
        fi
        # Cosmos-capable SGLang was developed against Torch 2.11. Resolve the
        # full CUDA-12.8 dependency set so NCCL/Triton stay ABI-compatible.
        uv pip install --torch-backend "$TORCH_BACKEND" --prerelease=allow --upgrade \
            "torch==$WORLDODYSSEY_SGLANG_TORCH_VERSION" \
            "torchvision==$WORLDODYSSEY_SGLANG_TORCHVISION_VERSION" \
            "torchaudio==$WORLDODYSSEY_SGLANG_TORCHAUDIO_VERSION"
        uv pip install --prerelease=allow \
            "transformers==$WORLDODYSSEY_SGLANG_TRANSFORMERS_VERSION" \
            "cosmos-guardrail==$WORLDODYSSEY_COSMOS3_GUARDRAIL_VERSION"
        # PyPI's matching kernel is CUDA 13. Use SGLang's official CUDA-12.9
        # wheel, which runs with the pinned CUDA-12.8 PyTorch runtime.
        uv pip install --reinstall \
            --index-url https://docs.sglang.ai/whl/cu129/ \
            "sglang-kernel==$WORLDODYSSEY_SGLANG_KERNEL_VERSION"
        # Rebuild VSA against the selected Torch ABI. --no-deps is essential:
        # its broad torch>=2.5 metadata would otherwise select CUDA 13.
        VSA_BUILD_ID="$VSA_SOURCE_REV:torch-$WORLDODYSSEY_SGLANG_TORCH_VERSION"
        VSA_BUILD_MARKER="$VENV_PATH/.worldodyssey-vsa-build"
        if [[ ! -f "$VSA_BUILD_MARKER" ]] && python - "$VSA_SOURCE_PATH" "$WORLDODYSSEY_SGLANG_TORCH_VERSION" <<'PY' >/dev/null 2>&1
from importlib import metadata
import json
from pathlib import Path
import sys

import torch
import vsa_cuda
from vsa import video_sparse_attn

assert torch.__version__.startswith(sys.argv[2] + "+") or torch.__version__ == sys.argv[2]
direct_url = json.loads(metadata.distribution("vsa").read_text("direct_url.json"))
expected = (Path(sys.argv[1]) / "csrc/attn/video_sparse_attn").resolve().as_uri()
assert direct_url["url"] == expected
PY
        then
            printf '%s\n' "$VSA_BUILD_ID" >"$VSA_BUILD_MARKER"
        fi
        if [[ ! -f "$VSA_BUILD_MARKER" || "$(<"$VSA_BUILD_MARKER")" != "$VSA_BUILD_ID" ]]; then
            CUDA_HOME="$CUDA_HOME_PATH" \
            PATH="$VENV_PATH/bin:$CUDA_HOME_PATH/bin:/usr/local/bin:/usr/bin:/bin" \
            CC=/usr/bin/gcc \
            CXX=/usr/bin/g++ \
            MAX_JOBS="${SGLANG_DIFFUSION_BUILD_JOBS:-16}" \
            uv pip install --no-deps --no-build-isolation --reinstall \
                "$VSA_SOURCE_PATH/csrc/attn/video_sparse_attn"
            printf '%s\n' "$VSA_BUILD_ID" >"$VSA_BUILD_MARKER"
        fi
        ;;
esac

if [[ "$PROFILE" == "unified" || "$PROFILE" == "hunyuan-fp8" ]]; then
    export PYTHONPATH="$SOURCE_PATH/python${PYTHONPATH:+:$PYTHONPATH}"
fi

python - "$PROFILE" "$SOURCE_PATH" "$SOURCE_REV" \
    "$WORLDODYSSEY_SGLANG_TORCH_VERSION" \
    "$WORLDODYSSEY_SGLANG_KERNEL_VERSION" \
    "$WORLDODYSSEY_SGLANG_TRANSFORMERS_VERSION" \
    "$WORLDODYSSEY_COSMOS3_GUARDRAIL_VERSION" <<'PY'
from importlib import metadata
from pathlib import Path
import sys

profile = sys.argv[1]
source_path = Path(sys.argv[2])
source_revision = sys.argv[3]
expected_versions = {
    "torch": sys.argv[4],
    "sglang-kernel": sys.argv[5],
    "transformers": sys.argv[6],
    "cosmos-guardrail": sys.argv[7],
}
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
        "nvidia-cuda-runtime",
        "nvidia-cuda-nvrtc",
        "nvidia-cublas",
        "nvidia-cudnn-cu13",
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
        "nvidia-cuda-runtime",
        "nvidia-cuda-nvrtc",
        "nvidia-cublas",
        "nvidia-cudnn-cu13",
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

if profile in {"unified", "hunyuan-fp8"}:
    for name, expected in expected_versions.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            raise SystemExit(f"Required package is not installed: {name}") from None
        if actual != expected and not actual.startswith(expected + "+"):
            raise SystemExit(f"{name} version mismatch: expected {expected}, got {actual}")

    import sglang
    from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
    from sglang.multimodal_gen.runtime.pipelines.cosmos3_pipeline import Cosmos3Pipeline
    import vsa_cuda
    from vsa import video_sparse_attn

    loaded = Path(sglang.__file__).resolve()
    expected_root = (source_path / "python").resolve()
    if not loaded.is_relative_to(expected_root):
        raise SystemExit(f"SGLang source overlay is not active: loaded {loaded}")
    print(f"sglang-source=={source_revision}")
    print(f"Cosmos support: {Cosmos3Config.__name__}, {Cosmos3Pipeline.__name__}")
    print(f"VSA support: {vsa_cuda.__file__}, {video_sparse_attn.__name__}")
PY

cat <<EOF

SGLang Diffusion profile installed:
  $PROFILE

Environment:
  $VENV_PATH

Pinned SGLang source:
  $SOURCE_PATH
  $SOURCE_REV

Pinned FastWan VSA source:
  $VSA_SOURCE_PATH
  $VSA_SOURCE_REV

Use these runtime guards before running SGLang Diffusion on this host:
  source "$VENV_PATH/bin/activate"
  export PATH="$VENV_PATH/bin:/usr/local/bin:/usr/bin:/bin"
  export CC=/usr/bin/gcc
  export CXX=/usr/bin/g++
  export CUDA_HOME="$CUDA_HOME_PATH"
  export PYTHONPATH="$SOURCE_PATH/python"

See references/sglang-diffusion.md for FastWan VSA commands and docs/video-backend-runbook.md for FP8 commands.
EOF
