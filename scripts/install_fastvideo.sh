#!/usr/bin/env bash
set -euo pipefail

echo "Warning: direct FastVideo is deprecated; supported FastWan inference uses .venv_sglang." >&2
echo "This installer is retained only for historical benchmark reproduction." >&2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../runtime-versions.env
set -a
source "$ROOT_DIR/runtime-versions.env"
set +a
VENV_PATH="${FASTVIDEO_VENV:-$ROOT_DIR/.venv_fastvideo}"
PYTHON_SPEC="${FASTVIDEO_PYTHON:-3.12}"
TORCH_BACKEND="${FASTVIDEO_TORCH_BACKEND:-cu128}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required to install FastVideo." >&2
    exit 1
fi

TARGET_PYTHON="$(uv python find "$PYTHON_SPEC")"
TARGET_VERSION="$($TARGET_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
INSTALLED_VERSION=""
if [[ -x "$VENV_PATH/bin/python" ]]; then
    INSTALLED_VERSION="$($VENV_PATH/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
fi

if [[ "$INSTALLED_VERSION" != "$TARGET_VERSION" ]]; then
    uv venv --clear --python "$TARGET_PYTHON" --seed "$VENV_PATH"
fi

uv pip install \
    --python "$VENV_PATH/bin/python" \
    --torch-backend "$TORCH_BACKEND" \
    "torch==$WORLDODYSSEY_FASTVIDEO_TORCH_VERSION" \
    "torchvision==$WORLDODYSSEY_FASTVIDEO_TORCHVISION_VERSION" \
    "torchaudio==$WORLDODYSSEY_FASTVIDEO_TORCHAUDIO_VERSION" \
    "diffusers==$WORLDODYSSEY_FASTVIDEO_DIFFUSERS_VERSION" \
    "huggingface-hub==$WORLDODYSSEY_FASTVIDEO_HUGGINGFACE_HUB_VERSION" \
    "hf-transfer==$WORLDODYSSEY_HF_TRANSFER_VERSION" \
    "fastvideo==$WORLDODYSSEY_FASTVIDEO_VERSION"

PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
PYTHONDONTWRITEBYTECODE=1 \
"$VENV_PATH/bin/python" - <<'PY'
from importlib import metadata
import os

import torch
from fastvideo import VideoGenerator

expected = {
    "torch": os.environ["WORLDODYSSEY_FASTVIDEO_TORCH_VERSION"],
    "torchvision": os.environ["WORLDODYSSEY_FASTVIDEO_TORCHVISION_VERSION"],
    "torchaudio": os.environ["WORLDODYSSEY_FASTVIDEO_TORCHAUDIO_VERSION"],
    "diffusers": os.environ["WORLDODYSSEY_FASTVIDEO_DIFFUSERS_VERSION"],
    "huggingface-hub": os.environ["WORLDODYSSEY_FASTVIDEO_HUGGINGFACE_HUB_VERSION"],
    "fastvideo": os.environ["WORLDODYSSEY_FASTVIDEO_VERSION"],
    "fastvideo-kernel": os.environ["WORLDODYSSEY_FASTVIDEO_KERNEL_VERSION"],
    "hf-transfer": os.environ["WORLDODYSSEY_HF_TRANSFER_VERSION"],
}
for package, version in expected.items():
    actual = metadata.version(package)
    if actual.split("+", 1)[0] != version:
        raise SystemExit(f"{package}: expected {version}, got {actual}")
    print(f"{package}=={actual}")

print(f"VideoGenerator={VideoGenerator.__name__}")
if torch.cuda.is_available():
    value = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    print(f"CUDA probe passed on {torch.cuda.get_device_name(0)}: {value.item():.0f}")
else:
    print("CUDA probe skipped: no visible NVIDIA GPU")
PY

cat <<EOF

FastVideo/FastWan runtime installed:
  $VENV_PATH

Run the FastWan example with:
  source "$VENV_PATH/bin/activate"
  python scripts/fastvideo_example.py
EOF
