# MiniMax-H3 through SGLang

MiniMax-H3 is isolated from the existing FastWan/Cosmos environment:

- `.venv_sglang` remains on the validated CUDA 12 SGLang source revision.
- `.venv_sglang_h3` uses pinned SGLang `0.5.18` with the CUDA-12.9 wheel override path.
- FL2VA listens on port `30010`; Ref2VA listens on port `30011`.
- The backend selects the server from the request mode and always uses H3's JSON `/v1/videos` contract.

The primary EgoAction path is FL2VA with one first-frame image plus text. Ref2VA is available when the image should
guide subject identity, style, or composition without being preserved as the exact first frame.

## Install

Install only the additional H3 runtime:

```bash
bash scripts/install_minimax_h3.sh
```

Or include it with the normal setup:

```bash
./install.sh --with-minimax-h3
```

The model itself should be downloaded before launch. Review and accept the MiniMax-H3 model license before use, then
download the primary FL2VA partition:

```bash
bash scripts/download_minimax_h3.sh fl2va
```

Expect the official BF16 FL2VA partition to occupy about 134 GiB. The advertised 33B parameter count describes the
H3 Omni-Transformer, while the self-contained checkpoint also includes the Qwen3-VL-32B-based H3 text encoder, the
Visual VAE, the Audio VAE, the processor, and the tokenizer. The Hub cache deduplicates blobs, so this footprint is the
required FL2VA payload rather than a duplicate download.

The H3 helper exports `HF_HOME` to `.cache/huggingface` under this repository by default. This avoids filling the
shared `/home` cache on machines with small user quotas. Override with `WORLDODYSSEY_HF_HOME=/path/to/cache` when a
different cache location is required. The helper also sets `HF_HUB_DISABLE_XET=1` by default because Xet-backed
reconstruction can fail on the shared filesystem; set `HF_HUB_DISABLE_XET=0` explicitly if that backend is desired.
The downloader uses four standard Hub workers by default; set `WORLDODYSSEY_HF_DOWNLOAD_MAX_WORKERS` explicitly to
adjust concurrency for a different network or filesystem.

The installer follows SGLang's documented CUDA-12 path: install the normal SGLang package first, then force-reinstall
the CUDA-12.9 PyTorch, `sglang-kernel`, and `sgl-deep-gemm` wheels. This keeps H3 usable on R570-series hosts whose
`nvidia-smi` reports CUDA 12.x, while avoiding a driver upgrade to R580/CUDA 13. CUDA 12 minor-version compatibility
allows CUDA 12.9 runtimes on R525+ drivers, subject to NVIDIA's normal caveats around newer driver-only features and
PTX.

The installer and server launchers also prepare the isolated H3 environment explicitly. In addition to putting
`.venv_sglang_h3/bin` on `PATH`, they prepend the venv's bundled NVIDIA wheel library directories to `LD_LIBRARY_PATH`.
The installer also fetches pinned static `ffmpeg` and `ffprobe` binaries and links them into the H3 venv. H3 uses
`ffprobe` to validate its joint audio/video file after denoising and decoding, so the launchers reject a runtime that
would otherwise fail only after completing generation. They activate the venv before invoking Python or SGLang.
Invoking `.venv_sglang_h3/bin/sglang` directly without this environment can make SGLang report that diffusion support
is unavailable even when the dependency extra and CUDA-12.9 kernel wheels are installed.
`scripts/install_minimax_h3.sh` and the normal server launchers fail fast if the media tools are absent or Torch cannot
initialize CUDA.

The default `memory` profile uses SGLang's lossless streaming configuration: layerwise-offload the DiT and text
encoder, prefetch one layer, keep zero DiT layers resident, and disable Torch compilation. Checkpoint size is therefore
not a VRAM requirement. SGLang documents the same BF16 baseline on a single 24 GB GPU at roughly 18 GB peak VRAM.
This repository's H3 task launchers default to four GPUs and TP4. They also enable SGLang's online AdaLN mode, which
rebuilds AdaLN outputs from the checkpoint for each request and leaves 24.2 GiB of AdaLN projection weights out of
each worker's resident state. FL2VA uses plan width 3; Ref2VA uses width 4. This preserves arbitrary unquantized
schedules while keeping TP4 startup under the current Slurm host-memory limit.

## FL2VA: text or endpoint images

Start the primary server:

```bash
bash scripts/run_minimax_h3_fl2va.sh
```

The default launch uses four GPUs with TP4, lossless layerwise offload, and online AdaLN. Override the topology
variables for a different validated multi-GPU layout, for example:

```bash
WORLDODYSSEY_MINIMAX_H3_NUM_GPUS=4 \
WORLDODYSSEY_MINIMAX_H3_TP_SIZE=2 \
WORLDODYSSEY_MINIMAX_H3_ULYSSES_DEGREE=2 \
WORLDODYSSEY_MINIMAX_H3_PERFORMANCE_MODE=speed \
bash scripts/run_minimax_h3_fl2va.sh
```

The task launcher still adds online AdaLN to this override. Pass
`--minimax-h3-adaln-online false` only on an allocation with enough host/device memory for fully resident AdaLN
projections.

The installer applies a version-specific SGLang 0.5.18 fix for online AdaLN slab overflow. Without it, changing the
inference-step schedule after another request can clear overlapping cached plans without rebuilding them, causing
`MiniMax H3 AdaLN cache does not cover the request timestep plan`. The patcher is idempotent and fails loudly if a
future SGLang release no longer matches the validated source context.

With the backend running, submit the checked-in one-image example:

```bash
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-i2v-minimax-h3-fl2va.yaml
```

The canonical `image_to_video` request becomes `task: fl2va` with a `keyframe` condition at frame index `0`. Set
`request.options.provider_options.minimax_h3.frame_index=-1` to treat the one image as the last frame. Supplying
`end_image_url` with the normal first image creates first-and-last-frame conditioning.

Text-only requests use `mode: text_to_video` and are sent as `task: t2va` to this same FL2VA deployment.

## Ref2VA: semantic image references

Start the reference server, optionally alongside FL2VA:

```bash
bash scripts/run_minimax_h3_ref2va.sh
```

Submit the checked-in one-reference example:

```bash
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-i2v-minimax-h3-ref2va.yaml
```

`mode: reference_to_video` becomes `task: ref2va`. The primary image and any `reference_image_urls` become ordered
`role: reference` conditions. This adapter currently scopes Ref2VA to one through nine image conditions; H3's video
and audio reference modes can be added later without changing FL2VA.

Prompts should name ordered materials as `<Picture 1>`, `<Picture 2>`, and so on.

## Run both deployments behind one backend

Start FL2VA and Ref2VA in separate terminals, then start the normal backend in a third terminal:

```bash
WORLDODYSSEY_MINIMAX_H3_FL2VA_BASE_URL=http://127.0.0.1:30010 \
WORLDODYSSEY_MINIMAX_H3_REF2VA_BASE_URL=http://127.0.0.1:30011 \
bash scripts/run_backend.sh
```

`scripts/run_backend.sh` activates the backend `.venv` before starting Python. Override the environment location with
`WORLDODYSSEY_BACKEND_VENV=/path/to/venv` when needed.

The backend routes `text_to_video` and `image_to_video` H3 jobs to FL2VA, and `reference_to_video` H3 jobs to Ref2VA.
FastWan and Cosmos requests continue to use `WORLDODYSSEY_SGLANG_BASE_URL` and their multipart request format.

## Request controls

Normal request fields:

- `options.duration`: 4–15 seconds; default `5`.
- `options.aspect_ratio`: for example `16:9`, or `auto` for FL2VA input geometry.
- `options.fps`: must be `24` when specified.
- `options.num_inference_steps`: default `50`.
- `options.seed`: optional deterministic seed.
- `options.generate_audio`: omit it or set `true`; H3 cannot disable its joint audio output.

H3-only fields live under `options.provider_options.minimax_h3`:

- `short_edge`: default `768`.
- `frame_index`: `0` or `-1` for a single FL2VA image.
- `flow_shift`: default `12.0`.
- `audio_flow_shift`: default `3.0`.
- `num_outputs_per_prompt`: default `1`.
- `quality`: `lossless` or `high`.
