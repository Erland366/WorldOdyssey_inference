# SGLang Diffusion Setup

This guide records the working local setup for SGLang Diffusion on the WorldOdyssey inference host. It keeps SGLang
outside the main `.venv` because the FastVideo/Diffusers stack and the SGLang Diffusion stack need different Torch and
kernel-wheel versions.

The local success path is:

- Dedicated venv: `.venv_sglangcuda12`
- `sglang[diffusion]==0.5.5`
- `torch==2.8.0+cu128`
- `sgl-kernel==0.3.16.post5`
- `vsa==0.0.4`
- `cuda-python==12.9.6`
- No `nvidia-*-cu13` runtime packages
- Explicit `CUDA_HOME` pointing at the venv's `nvidia` package
- Explicit system compiler and linker path so Triton does not pick Miniconda tools

Do not install current unpinned `sglang[diffusion]` into `.venv` on this host. The current latest wheel set can pull the
CUDA-13 `sglang-kernel` line, which imports CUDA 13 libraries that this driver stack cannot execute.

## Install

Run the installer from the repository root:

```bash
bash scripts/install_sglang_diffusion.sh
```

The script creates or updates `.venv_sglangcuda12` by default. To use a different environment path:

```bash
SGLANG_DIFFUSION_VENV=/abs/path/to/.venv_sglang bash scripts/install_sglang_diffusion.sh
```

The script intentionally uses `uv` only for virtual environment creation and dependency installation. It never uses
`uv run`.

## Runtime Environment

Before running `sglang`, activate the isolated environment and make the runtime paths explicit:

```bash
source .venv_sglangcuda12/bin/activate
export PATH="$PWD/.venv_sglangcuda12/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME="$PWD/.venv_sglangcuda12/lib/python3.12/site-packages/nvidia"
```

These guards are required for this host:

- `CUDA_HOME` prevents `sgl-kernel` from treating Miniconda as the CUDA installation.
- `PATH` puts `/usr/bin/ld` ahead of `/home/coder/miniconda3/bin/ld`.
- `CC` and `CXX` force Triton JIT compilation through the system compiler pair.

## Quick Probes

Run these after installation when validating a machine:

```bash
source .venv_sglangcuda12/bin/activate
export PATH="$PWD/.venv_sglangcuda12/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME="$PWD/.venv_sglangcuda12/lib/python3.12/site-packages/nvidia"

python - <<'PY'
from importlib import metadata
for name in (
    "torch",
    "sglang",
    "sgl-kernel",
    "vsa",
    "cuda-python",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-runtime-cu13",
):
    try:
        print(name, metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name, "not installed")
PY

python - <<'PY'
import torch

print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
assert torch.cuda.is_available()

x = torch.randn(1, 4, 5, 8, 8, device="cuda", dtype=torch.bfloat16)
conv = torch.nn.Conv3d(4, 8, 3, padding=1).cuda().bfloat16()
with torch.no_grad():
    y = conv(x)
torch.cuda.synchronize()
print("conv ok", tuple(y.shape))
PY

sglang generate --help
```

Expected local probes on May 15, 2026:

```text
torch 2.8.0+cu128
torch cuda 12.8
conv ok (1, 8, 5, 8, 8)
```

`nvidia-cuda-runtime-cu13` should be absent.

## FastWan VSA Validation

Use the repository long-running command contract. First run the foreground validation with a five-minute timeout:

```bash
source .venv_sglangcuda12/bin/activate
export PATH="$PWD/.venv_sglangcuda12/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME="$PWD/.venv_sglangcuda12/lib/python3.12/site-packages/nvidia"

mkdir -p artifacts/backend-videos/sglang-fastwan-vsa
rm artifacts/backend-videos/sglang-fastwan-vsa/fastwan-vsa-smoke.mp4 2>/dev/null || true

timeout 300s sglang generate \
  --model-path FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --attention-backend=video_sparse_attn \
  --VSA-sparsity=0.5 \
  --num-gpus=1 \
  --prompt "A calm ocean wave at sunrise" \
  --height=448 \
  --width=832 \
  --num-frames=61 \
  --num-inference-steps=3 \
  --seed=123 \
  --save-output \
  --output-path artifacts/backend-videos/sglang-fastwan-vsa \
  --output-file-name fastwan-vsa-smoke.mp4 \
  --log-level=debug

test -s artifacts/backend-videos/sglang-fastwan-vsa/fastwan-vsa-smoke.mp4
```

Validated result on May 15, 2026:

- Model: `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`
- Backend: `attention_backend=video_sparse_attn`
- VSA sparsity: `0.5`
- Shape: `448x832`, `61` frames, `3` denoising steps
- Output: `artifacts/backend-videos/sglang-fastwan-vsa/fastwan-vsa-smoke.mp4`
- Output size: about `835K`
- SGLang reported `Pixel data generated successfully in 20.81 seconds`

The log line `Selected attention backend: 'video_sparse_attn' not in supported attention backends ... Use fa3 as
default backend` can still appear for attention sites that do not support VSA, such as cross-attention. The Wan
transformer block path is still selected with `WanTransformerBlock_VSA`, and the pipeline logs `Using Video Sparse
Attention backend.` before denoising.

This SGLang release uses `--VSA-sparsity`. Newer SGLang docs describe `--attention-backend-config` for VSA, but that
flag shape is not present in `sglang==0.5.5`.

## Failure Modes

If `sglang generate --help` fails with `Could not find CUDA lib directory`, `CUDA_HOME` is probably resolving through
Miniconda. Set it to `.venv_sglangcuda12/lib/python3.12/site-packages/nvidia`.

If Triton fails with `/home/coder/miniconda3/bin/ld` and `unknown type [0x13] section '.relr.dyn'`, sanitize `PATH` so
`/usr/bin/ld` is selected before the Miniconda linker.

If VSA import fails with `No module named 'pytest'`, install `pytest` into `.venv_sglangcuda12`. The `vsa==0.0.4`
package imports `pytest` but does not declare it as a dependency.

If unpinned `sglang[diffusion]` installs `sglang-kernel` and `nvidia-*-cu13` packages, remove that environment and use
the pinned installer above. Do not try to fix this by adding CUDA 13 libraries to `LD_LIBRARY_PATH` on the current
driver.

## Upstream Documentation

The official SGLang Diffusion docs currently recommend `uv pip install "sglang[diffusion]" --prerelease=allow` for
standard NVIDIA installation, list `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` as VSA-compatible, and document
`video_sparse_attn` as the VSA backend. The local pinned setup deliberately diverges from the unpinned install command
because this host cannot use the current CUDA-13 SGLang kernel wheel stack.
