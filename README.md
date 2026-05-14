# WorldOdyssey Inference

Utilities for running and debugging Wan / WorldOdyssey inference experiments.

## Environment

Use the repository `.venv` directly. Do not use `uv run`.

```bash
cd /home/coder/Python_project/WorldOdyssey_inference
source .venv/bin/activate
python worldodyssey_inference/run_inference_multigpu.py
```

Install dependencies through the activated uv-managed environment:

```bash
source .venv/bin/activate
uv pip install <package>
```

## Tiny Wan Debug Pipelines

The real Wan models are too slow and memory-heavy for tight debugging loops. The script below creates randomly
initialized, tiny Wan artifacts that keep the same component classes and load path while using very small random
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

Create the Wan2.2 T2V-A14B-style local artifact. This recipe represents the high/low-noise expert layout as a
Diffusers `WanPipeline` with both `transformer` and `transformer_2` plus `boundary_ratio=0.875`.

```bash
source .venv/bin/activate
python scripts/create_tiny_wan_debug_pipeline.py --recipe wan2.2-t2v-a14b
```

Create the FastVideo FastWan2.1 DMD-style local artifact. This recipe saves the same tiny components in Diffusers
format, then patches `model_index.json` to `_class_name: "WanDMDPipeline"` so FastVideo can detect the DMD pipeline
fallback from a local path.

```bash
source .venv/bin/activate
python scripts/create_tiny_wan_debug_pipeline.py --recipe fastwan2.1-t2v-dmd
```

Create the FastVideo Wan2.1 VSA 14B 720P-style local artifact. This remains a `WanPipeline`, matching the upstream VSA
model metadata, but uses a tiny transformer head size that FastVideo's `VIDEO_SPARSE_ATTN` backend supports.
Although the metadata class is still `WanTransformer3DModel`, FastVideo swaps in its VSA transformer block at runtime
when `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`; the tiny fixture therefore patches the saved transformer
safetensors with `to_gate_compress` weights so FastVideo does not rely on missing-parameter fallback initialization.

```bash
source .venv/bin/activate
python scripts/create_tiny_wan_debug_pipeline.py --recipe wan2.1-vsa-t2v-14b-720p
```

Recreate an artifact after changing the fixture:

```bash
source .venv/bin/activate
python scripts/create_tiny_wan_debug_pipeline.py --recipe wan2.1-vsa-t2v-14b-720p --overwrite
```

Push the artifact to a public Hugging Face model repo. The script loads `.env`, derives the owner from `HF_TOKEN`, and
defaults to the recipe repo name unless `--repo-id` is supplied.

```bash
source .venv/bin/activate
python scripts/create_tiny_wan_debug_pipeline.py --recipe wan2.1-vsa-t2v-14b-720p --overwrite --push
```

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

The FastWan DMD recipe does not run the Diffusers `WanPipeline` smoke test because Diffusers does not provide
`WanDMDPipeline`. The script instead validates the saved `model_index.json` contract that FastVideo uses: required
components, `WanDMDPipeline` class name, and one `WanTransformer3DModel`.
The VSA recipe also uses FastVideo metadata validation because the success path is FastVideo loading with
`VIDEO_SPARSE_ATTN`; it checks that the tiny transformer's `attention_head_dim` is one of the head sizes supported by
that backend and that the FastVideo-only `to_gate_compress` tensors are present.

To add another tiny model later, add a `TinyWanRecipe` in `worldodyssey_inference/tiny_wan.py`. The CLI, local smoke
test, model card, and Hub upload path are shared by every recipe.

These artifacts are only for load-path, export, scheduler, CLI, and `device_map` debugging. They are random-weight and
must not be used for generation quality evaluation.

## Wan Backend Benchmarks

Use the benchmark harness to measure whether the real Wan/FastWan models fit on the local GPUs and to compare VRAM and
speed across Diffusers and FastVideo.

The primary matrix covers:

- `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` with Diffusers
- `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` with FastVideo
- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` with Diffusers
- `FastVideo/FastWan2.2-TI2V-5B-Diffusers` with FastVideo

Each case can run on one or two GPUs. Diffusers uses a single CUDA device for one-GPU runs and `device_map="balanced"`
for two-GPU runs. All child benchmark processes set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless the caller
already provides an allocator config. FastVideo additionally uses `VideoGenerator.from_pretrained(..., num_gpus=<N>)`
and sets `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`, `CC=/usr/bin/gcc`, and `CXX=/usr/bin/g++`. The allocator
setting matters on 24 GB cards; without it, some one-step fit checks can finish denoising and still OOM during VAE
decode.

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

Run the full two-stage matrix. The harness first runs one-step fit checks, then only benchmarks the cases that fit:

```bash
source .venv/bin/activate
python scripts/benchmark_wan_backends.py --stage all --gpus 1 2
```

Override the frame count for longer-video checks:

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

## Backend Slow Tests

Backend video generation tests live under `tests/backends/` and are marked `slow`. They generate videos into
`artifacts/backend-videos/`, which is ignored by git and intentionally persists after pytest exits so the outputs can be
inspected.

Run the backend examples explicitly:

```bash
source .venv/bin/activate
python -m pytest tests/backends -m slow
```

The Diffusers backend test uses `artifacts/tiny-wan2.1-t2v-debug` and rebuilds that local artifact if the tokenizer is
stale or incomplete. It no longer depends on an uploaded Hub copy for the smoke path.

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

Run the SGLang Diffusion native backend checks for the original tiny Wan and FastWan DMD artifacts:

```bash
source .venv/bin/activate
python -m pytest tests/backends/test_sglang_tiny_wan.py -m slow -s
```

SGLang Diffusion is optional because the current SGLang wheel stack is not dependency-compatible with the already
validated FastVideo stack in this single `.venv`. The SGLang test skips when `sglang` is not installed. If SGLang is
installed, the test runs the real `sglang generate` CLI and writes persistent videos to `artifacts/backend-videos/`.
For local tiny model paths, the test passes `--model-id Wan2.1-T2V-1.3B-Diffusers` or
`--model-id FastWan2.1-T2V-1.3B-Diffusers` so SGLang resolves the same native Wan/FastWan configuration it would use
for the official Hugging Face repos.

The FastVideo test sets `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++`, and `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`
inside the test before importing FastVideo. This avoids Triton picking the Miniconda compiler/linker from the shell.
The tiny FastWan test uses `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA` to keep the debugging fixture independent of the
custom sparse-attention kernels.
The tiny VSA test uses `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN` and is the acceptance check for that artifact.

On this host, `nvidia-smi` reports driver `550.127.08` with CUDA `12.4`. The previous FastVideo/Diffusers environment
used `torch==2.11.0+cu126`, which could import but failed during CUDA generation with
`CUDA driver version is insufficient for CUDA runtime version` or `CUDNN_STATUS_NOT_INITIALIZED`. The local `.venv`
now uses the CUDA 12.4 Torch stack described below.
Installing the current
`sglang[diffusion]` package pulled `sglang==0.5.11` and `sglang-kernel==0.4.2`, whose kernel extension needs CUDA 13
runtime libraries. Adding those libraries to `LD_LIBRARY_PATH` lets `sglang generate --help` start, but generation
fails during NCCL initialization with `CUDA driver version is insufficient for CUDA runtime version`. Use a CUDA-12
SGLang build/container or a newer NVIDIA driver before treating the SGLang slow test as an expected pass on this
machine.

The local FastVideo stack is pinned to a driver-compatible CUDA 12.4 Torch runtime:

- `torch==2.6.0+cu124`
- `torchvision==0.21.0+cu124`
- `torchaudio==2.6.0+cu124`
- `fastvideo==0.1.7` installed without allowing its dependencies to upgrade Torch

Use uv's PyTorch backend selector for the Torch stack, then install FastVideo without dependency resolution:

```bash
source .venv/bin/activate
uv pip install --torch-backend cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
uv pip install --no-deps fastvideo==0.1.7
```

`sitecustomize.py` applies a narrow Torch/FastVideo compatibility hook at interpreter startup. This is needed because
FastVideo worker subprocesses import FastVideo independently and `fastvideo==0.1.7` writes
`torch._dynamo.config.recompile_limit`, a key that does not exist in `torch==2.6.0`.

An isolated uv environment at `.venv_sglangcuda12` was tested for the CUDA-12 route. `torch==2.6.0+cu124` works there
with a minimal CUDA/cuDNN probe, but SGLang Diffusion starts at `sglang==0.5.5`, which pins `torch==2.8.0`, and PyTorch
does not publish `torch==2.8.0+cu124`. Forcing `sglang==0.5.5` onto `torch==2.6.0+cu124` with the available cu124
`sgl-kernel` wheels fails before the CLI starts because `sgl_kernel` cannot load `common_ops` due a torch C++ ABI
symbol mismatch. Treat `.venv_sglangcuda12` as a diagnostic environment, not a passing backend environment.
