---
name: sglang-hunyuan-fp8-diffusion
description: Run and debug HunyuanVideo ModelOpt FP8 diffusion on CUDA-12-constrained RTX 4090 hosts.
---

# SGLang Hunyuan FP8 Diffusion

## Purpose

Use this skill when running `hunyuanvideo-community/HunyuanVideo` with
`lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer` on a host that cannot use CUDA 13 packages.

This now runs from the unified `.venv_sglang` SGLang environment used by the other local diffusion models.

## Validated Host

The validated host had two NVIDIA GeForce RTX 4090 GPUs, compute capability 8.9, NVIDIA driver `550.127.08`, and about
24 GB VRAM per GPU.

FP8 primitives passed on this host with PyTorch cu128, and a native SGLang `/v1/videos` smoke request completed end to
end after the compatibility patches below.

## Known Working Package Set

```text
sglang==0.5.10.post1
torch==2.9.1+cu128
torchvision==0.24.1+cu128
sglang-kernel==0.4.1
flashinfer-python==0.6.7.post3
cuda-python==12.9.0
cuda-bindings==12.9.6
diffusers==0.37.0
transformers==5.3.0
tokenizers==0.22.1
nvidia-modelopt==0.44.0
accelerate==1.13.0
```

Do not use `sglang==0.5.11` on this host unless the CUDA 13 constraint is removed. It resolves into a CUDA 13 dependency
line that is incompatible with the driver constraint.

Install into the isolated environment with `uv pip install`, not plain `pip` and not `uv run`.

The project installer command is:

```bash
bash scripts/install_sglang_diffusion.sh
```

## Model Pair

The FP8 transformer repository is not a full pipeline. Use it as a transformer override:

```text
base model:       hunyuanvideo-community/HunyuanVideo
FP8 transformer:  lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer
```

The transformer snapshot is about 16 GB. If the cache only contains metadata, SGLang may fail with `no safetensors files
found`. Explicitly download the snapshot before retrying.

## Required Compatibility Patches

Stock `sglang==0.5.10.post1` does not fully support this FP8 diffusion model.

Backport SGLang main's diffusion ModelOpt FP8 quantizer into the isolated environment:

```text
sglang/multimodal_gen/runtime/layers/quantization/modelopt_fp8.py
```

Register `"modelopt"` in:

```text
sglang/multimodal_gen/runtime/layers/quantization/__init__.py
```

Expected quantization methods after registration:

```text
['modelopt', 'fp8', 'modelopt_fp4', 'modelslim']
```

Patch fused q/k/v scale loading narrowly. The observed failing parameter was:

```text
double_blocks.0.img_attn_qkv.input_scale
full_tensor.shape=(3,)
meta_sharded_param.shape=(1,)
param_cls=PerTensorScaleParameter
```

The working local patch reduced a fused `PerTensorScaleParameter` load to one scalar with `loaded_weight.max()` when the
runtime parameter has one element and the checkpoint tensor has multiple elements. This matches the backported ModelOpt
FP8 quantizer's later max-scale behavior for static per-tensor FP8 scales.

Apply or refresh these patches with:

```bash
source .venv_sglang/bin/activate
python scripts/patch_sglang_hunyuan_fp8.py --venv .venv_sglang
```

Layerwise offload can materialize ModelOpt FP8 weights with row-major strides even though `sgl_kernel.fp8_scaled_mm`
requires the transposed FP8 weight to be column-major. The local `modelopt_fp8.py` shim repairs this before the GEMM by
replacing row-major 2D FP8 weights with `weight.t().contiguous().t()`.

The patch script also extends `SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1` to SGLang-Diffusion's fused residual norm
wrappers. The project `memory` offload preset defaults that env var to `1` so Hunyuan FP8 avoids CuTe DSL fused norm
kernels that can fail with CUDA misaligned-address errors on this RTX 4090 host.

The project `memory` preset also defaults `USE_TRITON_W8A8_FP8_KERNEL=1`, `SGLANG_DISABLE_FLASHINFER_ROPE=1`, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, then passes smaller VAE tile args:
`--vae-config.tile-sample-min-height 128`, `--vae-config.tile-sample-min-width 128`,
`--vae-config.tile-sample-stride-height 96`, `--vae-config.tile-sample-stride-width 96`,
`--vae-config.tile-sample-min-num-frames 8`, and `--vae-config.tile-sample-stride-num-frames 4`. These settings were
needed for the larger WorldOdyssey `960x544`, 20-frame visual config: Triton avoided the CUTLASS FP8 scaled-mm failure,
smaller VAE tiles avoided a post-denoising Hunyuan VAE `conv3d` OOM, and the FlashInfer RoPE guard lets Wan InP use
SGLang's Triton RoPE fallback when `nvcc` is absent from the uv runtime.

## Startup Command

Use foreground validation first. If the server reaches a healthy state, restart it in tmux for long-lived use.

```bash
WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory \
WORLDODYSSEY_SGLANG_LOG_LEVEL=debug \
WORLDODYSSEY_SGLANG_NUM_GPUS=2 \
WORLDODYSSEY_SGLANG_TP_SIZE=1 \
WORLDODYSSEY_SGLANG_SP_DEGREE=2 \
bash scripts/serve_sglang_diffusion.sh hunyuanvideo-community/HunyuanVideo \
  --transformer-path lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer
```

Successful startup should include:

```text
Loaded model with 12.82B parameters
Loaded transformer: HunyuanVideoTransformer3DModel
Uvicorn running on http://127.0.0.1:30000
```

Idle memory after a smoke run was about 20.4 GB used per RTX 4090, so production resolutions may still be tight.

## Native Smoke Test

The newer SGLang `/v1/videos` endpoint is multipart/form-data. Use a tiny smoke request before any full-resolution run:

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

Expected result:

```text
status: completed
progress: 100
```

The validated smoke output was:

```text
artifacts/video-backend/75a30c04-59f1-4255-ba8c-6ac9aee91d1b.mp4
```

The provider-neutral FastAPI app was also validated against the FP8 server on 2026-05-18. Job
`vid_20260518T065703Z_04fe6059` wrote:

```text
artifacts/video-backend-api-smoke/videos/vid_20260518T065703Z_04fe6059/output.mp4
```

The larger WorldOdyssey move-bookmark visual config was validated on 2026-05-18 with the `memory` preset. Job
`vid_20260518T113402Z_9c852dfc` wrote:

```text
artifacts/video-backend/worldodyssey-move-bookmark-hunyuan-fp8-visual.mp4
```

`128x128` is intentionally below supported quality resolutions. SGLang warns that output quality may suffer; this is fine
for smoke testing the FP8 path.

## Failure Modes

If CLIP tokenizer loading fails with:

```text
RobertaProcessing.__new__() got an unexpected keyword argument 'cls'
```

pin `tokenizers==0.22.1`.

If the transformer loader fails with:

```text
no safetensors files found
```

explicitly download `lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer`; the local cache may only contain metadata.

If SGLang fails with:

```text
Invalid quantization method: modelopt
```

the diffusion ModelOpt FP8 quantizer is missing or not registered.

If SGLang fails on `double_blocks.*.img_attn_qkv.input_scale` with `(3,)` versus `(1,)`, the fused scale loader patch is
missing.

If SGLang fails under `--dit-layerwise-offload` with:

```text
RuntimeError: mat_b must be a column major tensor
```

the ModelOpt FP8 column-major repair is missing from the installed environment. Rerun
`python scripts/patch_sglang_hunyuan_fp8.py --venv .venv_sglang`, restart SGLang, then retry the request.

If SGLang fails under `--dit-layerwise-offload` with:

```text
RuntimeError: CUDA Error: cudaErrorMisalignedAddress
```

from `cutedsl/scale_residual_norm_scale_shift.py`, the deterministic norm fallback patch is missing or the server was
started with `SGLANG_ENABLE_DETERMINISTIC_INFERENCE=0`.

If GPU memory is near the limit at idle, avoid full-resolution runs until a lower-step supported-resolution probe passes.

If denoising completes but Hunyuan VAE decode fails with a `conv3d` OOM, restart with
`WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory` so the smaller VAE tile defaults are active, or pass even smaller trailing
`--vae-config.*` tile values.

## Project Backend

Start the provider-neutral backend:

```bash
export WORLDODYSSEY_SGLANG_BASE_URL=http://127.0.0.1:30000
source .venv/bin/activate
python scripts/serve_video_backend.py --host 127.0.0.1 --port 8000
```

The FastAPI request remains JSON. The local provider converts it to native SGLang multipart and forwards
`negative_prompt`, `num_inference_steps`, `seed`, `guidance_scale`, scalar `provider_options.request_fields`, and
structured `provider_options.extra_body`.
