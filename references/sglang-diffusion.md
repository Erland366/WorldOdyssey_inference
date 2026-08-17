# SGLang Diffusion Setup

This guide records the working local setup for SGLang Diffusion on the WorldOdyssey inference host. The lightweight
backend stays in `.venv`; every supported local GPU model, including FastWan and Cosmos 3, uses `.venv_sglang`.

The local unified success path is:

- Dedicated venv: `.venv_sglang`
- stable compiled base: `sglang[diffusion]==0.5.10.post1`
- Cosmos-capable SGLang source overlay pinned by `WORLDODYSSEY_SGLANG_SOURCE_REV`
- `torch==2.11.0+cu128`
- `sglang-kernel==0.4.3+cu129`
- `cuda-python==12.9.0`
- `cosmos-guardrail==0.3.1`
- VSA built locally from FastVideo source revision `055e52e5ea44b30cf982a3f248baa36ef386a39c`
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

The script creates or updates `.venv_sglang`, checks out the exact SGLang source revision under `.deps/sglang`, and
checks out the VSA source revision recorded by `WORLDODYSSEY_VSA_SOURCE_REV` under `.deps/fastvideo-vsa`. It builds
that VSA extension locally against the pinned Torch ABI. Do not replace it with the PyPI VSA wheel, which targets an
older Torch ABI and is incompatible with this unified environment. To use different managed paths:

```bash
SGLANG_DIFFUSION_VENV=/abs/path/to/.venv_sglang \
WORLDODYSSEY_SGLANG_SOURCE=/abs/path/to/sglang-source \
bash scripts/install_sglang_diffusion.sh
```

The script intentionally uses `uv` only for virtual environment creation and dependency installation. It never uses
`uv run`.

## Runtime Environment

Before running `sglang`, activate the isolated environment and make the runtime paths explicit:

```bash
source .venv_sglang/bin/activate
export PATH="$PWD/.venv_sglang/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$PWD/.deps/sglang/python"
```

These guards are required for this host:

- `CUDA_HOME` points JIT compilation at the complete CUDA 12.8 toolkit, including `nvcc`.
- `PATH` puts `/usr/bin/ld` ahead of `/home/coder/miniconda3/bin/ld`.
- `CC` and `CXX` force Triton JIT compilation through the system compiler pair.

## Quick Probes

Run these after installation when validating a machine:

```bash
source .venv_sglang/bin/activate
export PATH="$PWD/.venv_sglang/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="$PWD/.deps/sglang/python"

python - <<'PY'
from importlib import metadata
for name in (
    "torch",
    "sglang",
    "sglang-kernel",
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

sglang serve --help
```

Expected local probes for the unified runtime:

```text
torch 2.11.0+cu128
torch cuda 12.8
conv ok (1, 8, 5, 8, 8)
```

`nvidia-cuda-runtime-cu13` should be absent.

## FastWan VSA Validation

Use the repository long-running command contract. First run the foreground validation with a five-minute timeout:

```bash
source .venv_sglang/bin/activate
export PATH="$PWD/.venv_sglang/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME="$PWD/.venv_sglang/lib/python3.12/site-packages/nvidia"

timeout 300s bash scripts/serve_sglang_diffusion.sh FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --attention-backend video_sparse_attn \
  --VSA-sparsity 0.5
```

Validated end-to-end result on August 17, 2026:

- Model: `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`
- Backend: `attention_backend=video_sparse_attn`
- VSA sparsity: `0.5`
- Request: `320x192`, 5 frames, 1 inference step, 16 fps
- Provider-neutral API elapsed time: 2.1697 seconds
- Validated output: H.264, `320x192`, 16 fps, 5 decoded frames
- Artifact: `artifacts/backend-videos/fastwan-sglang-unified-smoke.mp4`

The log line `Selected attention backend: 'video_sparse_attn' not in supported attention backends ... Use fa3 as
default backend` can still appear for attention sites that do not support VSA, such as cross-attention. The Wan
transformer block path is still selected with `WanTransformerBlock_VSA`, and the pipeline logs `Using Video Sparse
Attention backend.` before denoising.

The local launcher accepts the established `--VSA-sparsity` flag and translates it to the pinned native server's
structured `--attention-backend-config` argument.

## Memory Offload

Use `WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory` when a native SGLang Diffusion video model does not fit comfortably in
VRAM:

```bash
WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory \
WORLDODYSSEY_SGLANG_LOG_LEVEL=debug \
WORLDODYSSEY_SGLANG_NUM_GPUS=1 \
bash scripts/serve_sglang_diffusion.sh weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers
```

The preset is explicit and default-off. It appends these native SGLang-Diffusion flags before any user-supplied trailing
flags:

```text
--dit-layerwise-offload
--dit-cpu-offload false
--dit-offload-prefetch-size 0
--text-encoder-cpu-offload
--image-encoder-cpu-offload
--vae-cpu-offload
--pin-cpu-memory
--vae-tiling
--vae-slicing
--vae-config.tile-sample-min-height 128
--vae-config.tile-sample-min-width 128
--vae-config.tile-sample-stride-height 96
--vae-config.tile-sample-stride-width 96
--vae-config.tile-sample-min-num-frames 8
--vae-config.tile-sample-stride-num-frames 4
```

Use `--dit-offload-prefetch-size 0` first for the lowest VRAM pressure. If startup and a smoke request are stable but
latency is too high, pass a later override after the model id, for example `--dit-offload-prefetch-size 0.1`. Do not use
this preset with `SGLANG_CACHE_DIT_ENABLED=true`; the launcher rejects that combination because SGLang-Diffusion treats
Cache-DiT and layerwise DiT offload as incompatible.

The `memory` preset defaults `SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1` unless it is already set. The local patch extends
that deterministic flag to SGLang-Diffusion's fused residual norm wrappers so the low-memory path uses native norm
fallbacks instead of CuTe DSL fused norm kernels on this RTX 4090 host.

The preset also defaults `USE_TRITON_W8A8_FP8_KERNEL=1` unless it is already set. On the validated RTX 4090 host,
Hunyuan ModelOpt FP8 denoising with layerwise offload reached SGLang's CUTLASS W8A8 FP8 branch and failed with a
`cutlass::Status::kSuccess` assertion. The Triton W8A8 FP8 branch completed denoising. The smaller VAE tile defaults
above avoid a post-denoising Hunyuan VAE `conv3d` OOM at `960x544`, 20 frames, 6 steps.

The preset also defaults `SGLANG_DISABLE_FLASHINFER_ROPE=1` unless it is already set. The local patch adds that env
guard to SGLang's RoPE utility; Wan InP otherwise reaches FlashInfer's RoPE JIT and fails in the uv runtime because
`flashinfer` tries to execute an absent `nvidia/bin/nvcc`. With the guard enabled, SGLang takes its Triton RoPE
fallback and the validated `worldodyssey-inputs-batch-t2v-wan-InP.yaml` request completes.

Hunyuan FP8 with layerwise offload needs the local ModelOpt FP8 column-major repair from
`patches/sglang_hunyuan_fp8/modelopt_fp8.py`. SGLang's layerwise offload restores FP8 weights from flat CPU buffers; if
the transposed FP8 weight becomes row-major, `sgl_kernel.fp8_scaled_mm` fails with:

```text
RuntimeError: mat_b must be a column major tensor
```

Rerun `python scripts/patch_sglang_hunyuan_fp8.py --venv .venv_sglang` after changing or recreating the SGLang
environment.

If Hunyuan FP8 with layerwise offload fails with:

```text
RuntimeError: CUDA Error: cudaErrorMisalignedAddress
```

from `cutedsl/scale_residual_norm_scale_shift.py`, the deterministic norm fallback patch is missing or the server was
started with `SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0`. Reapply the patch and restart SGLang.

## Hunyuan FP8 Validation

Start the native SGLang server:

```bash
WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory \
WORLDODYSSEY_SGLANG_LOG_LEVEL=debug \
WORLDODYSSEY_SGLANG_NUM_GPUS=2 \
WORLDODYSSEY_SGLANG_TP_SIZE=1 \
WORLDODYSSEY_SGLANG_SP_DEGREE=2 \
bash scripts/serve_sglang_diffusion.sh hunyuanvideo-community/HunyuanVideo \
  --transformer-path lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer
```

The FP8 transformer repository is not a full pipeline. Requests use the base model id
`hunyuanvideo-community/HunyuanVideo`; the transformer override is supplied at server launch.

Tiny native smoke request:

```bash
curl -X POST http://127.0.0.1:30000/v1/videos \
  -F 'prompt=A tiny red cube rotating on a plain gray background' \
  -F 'model=hunyuanvideo-community/HunyuanVideo' \
  -F 'size=128x128' \
  -F 'fps=8' \
  -F 'num_frames=5' \
  -F 'num_inference_steps=1' \
  -F 'seed=1'
```

Validated locally on 2026-05-18 with `sglang==0.5.10.post1`; the output was
`artifacts/video-backend/75a30c04-59f1-4255-ba8c-6ac9aee91d1b.mp4`.

The same running FP8 server was validated through the provider-neutral FastAPI app on 2026-05-18; job
`vid_20260518T065703Z_04fe6059` wrote
`artifacts/video-backend-api-smoke/videos/vid_20260518T065703Z_04fe6059/output.mp4`.

The larger WorldOdyssey move-bookmark visual config was validated end to end on 2026-05-18 with the `memory` preset;
job `vid_20260518T113402Z_9c852dfc` wrote
`artifacts/video-backend/worldodyssey-move-bookmark-hunyuan-fp8-visual.mp4`.

## Failure Modes

If `sglang serve --help` fails with `Could not find CUDA lib directory`, `CUDA_HOME` is probably resolving through
Miniconda. Set it to `.venv_sglang/lib/python3.12/site-packages/nvidia`.

If Triton fails with `/home/coder/miniconda3/bin/ld` and `unknown type [0x13] section '.relr.dyn'`, sanitize `PATH` so
`/usr/bin/ld` is selected before the Miniconda linker.

If unpinned `sglang[diffusion]` installs `sglang-kernel` and `nvidia-*-cu13` packages, remove that environment and use
the pinned installer above. Do not try to fix this by adding CUDA 13 libraries to `LD_LIBRARY_PATH` on the current
driver.

For the unified profile, `sglang-kernel==0.4.3+cu129` is expected. CUDA 13 runtime packages are still not expected.

If Hunyuan FP8 fails with `Invalid quantization method: modelopt`, rerun:

```bash
source .venv_sglang/bin/activate
python scripts/patch_sglang_hunyuan_fp8.py --venv .venv_sglang
```

## Upstream Documentation

The official SGLang Diffusion docs currently recommend `uv pip install "sglang[diffusion]" --prerelease=allow` for
standard NVIDIA installation, list `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` as VSA-compatible, and document
`video_sparse_attn` as the VSA backend. The local pinned setup deliberately diverges from the unpinned install command
because this host cannot use the current CUDA-13 SGLang kernel wheel stack.
