# Tiny Models And Diffusers

This document keeps the non-SGLang-pipeline material out of the main README: tiny Wan fixtures, Diffusers/FastVideo
benchmarks, and backend slow tests. These workflows are useful for debugging load paths and comparing model backends,
but they are not the primary WorldOdyssey SGLang API workflow.

## Tiny Wan Debug Pipelines

The real Wan models are too slow and memory-heavy for tight debugging loops. The tiny pipeline script creates
randomly initialized artifacts that preserve component classes, metadata, and load paths while using very small random
weights.

List supported recipes:

```bash
source .venv/bin/activate
python scripts/create_tiny_wan_debug_pipeline.py --list-recipes
```

Create the default Wan2.1 local artifact:

```bash
source .venv/bin/activate
python scripts/create_tiny_wan_debug_pipeline.py
```

Create the Wan2.2 T2V-A14B-style local artifact:

```bash
source .venv/bin/activate
python scripts/create_tiny_wan_debug_pipeline.py --recipe wan2.2-t2v-a14b
```

Create the FastVideo FastWan2.1 DMD-style artifact:

```bash
source .venv/bin/activate
python scripts/create_tiny_wan_debug_pipeline.py --recipe fastwan2.1-t2v-dmd
```

Create the FastVideo Wan2.1 VSA 14B 720P-style artifact:

```bash
source .venv/bin/activate
python scripts/create_tiny_wan_debug_pipeline.py --recipe wan2.1-vsa-t2v-14b-720p
```

Recreate an artifact after changing a recipe:

```bash
source .venv/bin/activate
python scripts/create_tiny_wan_debug_pipeline.py --recipe wan2.1-vsa-t2v-14b-720p --overwrite
```

Push an artifact to a public Hugging Face model repo. The script loads `.env`, derives the owner from `HF_TOKEN`, and
defaults to the recipe repo name unless `--repo-id` is supplied.

```bash
source .venv/bin/activate
python scripts/create_tiny_wan_debug_pipeline.py --recipe wan2.1-vsa-t2v-14b-720p --overwrite --push
```

These artifacts are only for load-path, export, scheduler, CLI, and `device_map` debugging. They are random-weight and
must not be used for generation quality evaluation.

## Tiny Artifact Usage

Use uploaded `WanPipeline` artifacts with Diffusers:

```python
from diffusers import WanPipeline

pipe = WanPipeline.from_pretrained("YOUR_HF_USERNAME/tiny-wan2.2-t2v-a14b-debug")
pipe.set_progress_bar_config(disable=True)
frames = pipe(
    prompt="debug prompt",
    height=64,
    width=64,
    num_frames=5,
    num_inference_steps=1,
    guidance_scale=1.0,
    max_sequence_length=8,
).frames[0]
```

Use the FastWan DMD artifact with FastVideo:

```python
import os
from fastvideo import VideoGenerator

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
generator = VideoGenerator.from_pretrained(
    "YOUR_HF_USERNAME/tiny-fastwan2.1-t2v-dmd-debug",
    num_gpus=1,
)
try:
    generator.generate_video(
        "debug prompt",
        output_path="my_videos/",
        save_video=True,
    )
finally:
    generator.shutdown()
```

Use the VSA artifact with FastVideo:

```python
import os
from fastvideo import VideoGenerator

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
generator = VideoGenerator.from_pretrained(
    "YOUR_HF_USERNAME/tiny-wan2.1-vsa-t2v-14b-720p-debug",
    num_gpus=1,
    pipeline_config={"flow_shift": 5.0},
    VSA_sparsity=0.5,
)
try:
    generator.generate_video(
        "debug prompt",
        output_path="my_videos/",
        save_video=True,
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=3,
        guidance_scale=1.0,
    )
finally:
    generator.shutdown()
```

The FastWan DMD recipe does not run a Diffusers `WanPipeline` smoke test because Diffusers does not provide
`WanDMDPipeline`. The script instead validates the `model_index.json` contract that FastVideo uses.

The VSA recipe uses FastVideo metadata validation because the success path is FastVideo loading with
`VIDEO_SPARSE_ATTN`. It checks that the tiny transformer's `attention_head_dim` is supported by that backend and that
FastVideo-only `to_gate_compress` tensors are present.

To add another tiny model later, add a `TinyWanRecipe` in `worldodyssey_inference/tiny_wan.py`. The CLI, local smoke
test, model card, and Hub upload path are shared by every recipe.

## Tiny Wan Through The Backend

The provider-neutral backend also accepts the public Hugging Face debug model
`Erland/tiny-wan2.1-t2v-debug` for low-memory T2V batch checks. Its model metadata marks it as a Diffusers
`WanPipeline`, `tiny-random`, and `debug` text-to-video model, so the backend treats it as a non-VSA tiny path.

The backend path still requires a persistent native SGLang server. Start SGLang with the tiny model before submitting
the batch:

```bash
WORLDODYSSEY_SGLANG_NUM_GPUS=1 \
bash scripts/serve_sglang_diffusion.sh Erland/tiny-wan2.1-t2v-debug
```

Then start the provider-neutral backend with:

```bash
export WORLDODYSSEY_SGLANG_BASE_URL=http://127.0.0.1:30000
source .venv/bin/activate
python scripts/serve_video_backend.py --host 127.0.0.1 --port 8000
```

The backend forwards `request.model` to native SGLang, does not stage or patch the model, and does not fall back to
one-shot generation. If the native SGLang Diffusion server rejects a tiny artifact, fix the artifact or server launch
path directly and keep the backend pointed at the working SGLang server.

Use the checked-in batch config:

```bash
source .venv/bin/activate
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml --dry-run
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml
```

Keep `attention_backend` and `vsa_sparsity` out of the YAML request. Configure attention backends only on the native
SGLang server command.

## Wan Backend Benchmarks

Use the benchmark harness to measure whether real Wan/FastWan models fit on the local GPUs and to compare VRAM and
speed across Diffusers and FastVideo.

The primary matrix covers:

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` with Diffusers
- `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` with FastVideo
- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` with Diffusers
- `FastVideo/FastWan2.2-TI2V-5B-Diffusers` with FastVideo

Each case can run on one or two GPUs. Diffusers uses a single CUDA device for one-GPU runs and
`device_map="balanced"` for two-GPU runs. Benchmark workers set
`SGLANG_DISABLE_FLASHINFER_ROPE=1` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless the caller already
provides those env vars. The FlashInfer guard keeps Wan InP on SGLang's Triton RoPE fallback when the uv runtime has no
`nvcc` for FlashInfer's JIT cache build.

Run a one-step fit check for the FastVideo 5B model on one GPU:

```bash
source .venv/bin/activate
python scripts/benchmark_wan_backends.py --stage fit --case fastvideo-5b --gpus 1
```

On RTX 4090s, the default `704x1280x121` FastVideo 5B fit check does not fit on one GPU. The validated local path is
two GPUs:

```bash
source .venv/bin/activate
python scripts/benchmark_wan_backends.py --stage fit --case fastvideo-5b --gpus 2 --no-save-video
```

Save the passing one-step 5B fit output video:

```bash
source .venv/bin/activate
python scripts/benchmark_wan_backends.py --stage fit --case fastvideo-5b --gpus 2 --save-fit-video
```

Run the full two-stage matrix. The harness first runs one-step fit checks, then only benchmarks the cases that fit:

```bash
source .venv/bin/activate
python scripts/benchmark_wan_backends.py --stage all --gpus 1 2
```

Override frame count for longer-video checks:

```bash
source .venv/bin/activate
python scripts/benchmark_wan_backends.py --stage all --case diffusers-1.3b fastvideo-1.3b --gpus 1 2 --num-frames 121
```

Results are written under `benchmark_results/wan_backend_benchmarks/<timestamp>/`:

- `preflight.json`: package, GPU, Diffusers import, FastVideo import, and tiny CUDA/cuDNN probe results
- `results.jsonl`: one JSON record per benchmark cell
- `summary.md`: compact table of status, load time, generation time, peak VRAM, and video paths
- `logs/*.log`: stdout/stderr for each isolated worker process

Generated benchmark videos are written under `artifacts/benchmark-videos/<timestamp>/` and intentionally persist for
inspection. Both `benchmark_results/` and `artifacts/` are ignored by git.

## Measured Benchmark Results

Measured one-step 5B fit checks on this machine:

| Case | GPUs | Status | Notes |
|---|---:|---|---|
| `fastvideo-5b` | 1 | OOM | Reaches VAE decode, then exceeds 24 GB. |
| `fastvideo-5b` | 2 | Passed | Requires the harness allocator setting; peak was about 24.2 GB and 24.1 GB. |
| `diffusers-5b` | 1 | OOM | Fails during the first denoising step. |
| `diffusers-5b` | 2 | OOM | Fails during VAE decode, even with expandable CUDA segments. |

Measured 1.3B benchmark results on May 11, 2026:

| Case | GPUs | Status | Load s | Generate s | Peak VRAM |
|---|---:|---|---:|---:|---|
| `diffusers-1.3b` | 1 | Passed | 4.13 | 259.55 | GPU 0: `23652 MiB` |
| `diffusers-1.3b` | 2 | Passed | 5.33 | 260.61 | GPU 0: `11694 MiB`, GPU 1: `14052 MiB` |
| `fastvideo-1.3b` | 1 | Passed | 41.13 | 26.83 | GPU 0: `19232 MiB` |
| `fastvideo-1.3b` | 2 | Passed | 43.12 | 30.78 | GPU 0: `19825 MiB`, GPU 1: `18986 MiB` |

The run summary is in `benchmark_results/wan_backend_benchmarks/20260511T091815Z/summary.md`, and generated videos are
in `artifacts/benchmark-videos/20260511T091815Z/`.

Measured 121-frame 1.3B benchmark results on May 11, 2026:

| Case | GPUs | Status | Load s | Generate s | Peak VRAM |
|---|---:|---|---:|---:|---|
| `diffusers-1.3b` | 1 | Passed | 4.03 | 475.28 | GPU 0: `24192 MiB` |
| `diffusers-1.3b` | 2 | Passed | 5.30 | 478.32 | GPU 0: `11694 MiB`, GPU 1: `14896 MiB` |
| `fastvideo-1.3b` | 1 | Passed | 40.50 | 44.86 | GPU 0: `20074 MiB` |
| `fastvideo-1.3b` | 2 | Passed | 42.81 | 45.64 | GPU 0: `20048 MiB`, GPU 1: `19660 MiB` |

The 121-frame run summary is in `benchmark_results/wan_backend_benchmarks/20260511T130630Z/summary.md`, and generated
videos are in `artifacts/benchmark-videos/20260511T130630Z/`.

## Backend Slow Tests

Backend video generation tests live under `tests/backends/` and are marked `slow`. They generate videos into
`artifacts/backend-videos/`, which is ignored by git and intentionally persists after pytest exits so outputs can be
inspected.

Run the backend examples explicitly:

```bash
source .venv/bin/activate
python -m pytest tests/backends -m slow
```

Run only the Diffusers tiny backend example:

```bash
source .venv/bin/activate
python -m pytest tests/backends/test_diffusers_example.py -m slow
```

Run only the FastVideo backend example:

```bash
source .venv/bin/activate
python -m pytest tests/backends/test_fastvideo_example.py -m slow
```

Run only the tiny FastWan DMD FastVideo example:

```bash
source .venv/bin/activate
python -m pytest tests/backends/test_fastvideo_tiny_fastwan.py -m slow
```

Run only the tiny VSA FastVideo example:

```bash
source .venv/bin/activate
python -m pytest tests/backends/test_fastvideo_tiny_vsa.py -m slow
```

Run only the low-level SGLang CLI checks for the original tiny Wan and FastWan DMD artifacts:

```bash
source .venv/bin/activate
python -m pytest tests/backends/test_sglang_tiny_wan.py -m slow -s
```

SGLang Diffusion is optional in the main `.venv` because the current SGLang wheel stack is not dependency-compatible
with the already validated FastVideo/Diffusers stack in that single environment. The SGLang test skips when `sglang` is
not installed. If SGLang is installed in that environment, the test runs the real `sglang generate` CLI and writes
persistent videos to `artifacts/backend-videos/`. This is not the provider-neutral backend path.

## Local FastVideo Stack Notes

The local FastVideo stack is pinned to a driver-compatible CUDA 12.4 Torch runtime:

- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`
- `fastvideo==0.1.7` installed without allowing dependencies to upgrade Torch

Use uv's PyTorch backend selector for the Torch stack, then install FastVideo without dependency resolution:

```bash
source .venv/bin/activate
uv pip install --torch-backend cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
uv pip install --no-deps fastvideo==0.1.7
```

`sitecustomize.py` applies a narrow Torch/FastVideo compatibility hook at interpreter startup. This is needed because
FastVideo worker subprocesses import FastVideo independently and `fastvideo==0.1.7` writes
`torch._dynamo.config.recompile_limit`, a key that does not exist in `torch==2.6.0`.

The failed CUDA-12.4 SGLang experiment is documented in `references/troubleshooting.md`: forcing `sglang==0.5.5` onto
`torch==2.6.0+cu124` fails before the CLI starts because `sgl_kernel` cannot load `common_ops` due to a torch C++ ABI
symbol mismatch. The working SGLang path is the isolated `.venv_sglang` stack described in
`references/sglang-diffusion.md`.
