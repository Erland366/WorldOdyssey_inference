# WorldOdyssey Inference

Provider-neutral SGLang Diffusion pipeline for WorldOdyssey video inference.

The primary local path is:

- FastAPI backend in the main `.venv`
- isolated SGLang Diffusion runtime in `.venv_sglang`
- local provider id: `sglang`
- native SGLang server launcher: `scripts/serve_sglang_diffusion.sh`
- optional launcher memory offload preset: `WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory`
- validated T2V model: `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`
- validated T2V attention backend: `video_sparse_attn` with `vsa_sparsity=0.5`
- validated FP8 model pair: `hunyuanvideo-community/HunyuanVideo` with
  `lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer`
- default I2V model: `FastVideo/FastWan2.2-TI2V-5B-Diffusers`
- tiny T2V debug model: `Erland/tiny-wan2.1-t2v-debug`

There is no local one-shot generation fallback. The backend requires an already-running native SGLang Diffusion server
and calls SGLang's `/v1/videos` API. The normal server path uses one native multipart SGLang server for FastWan,
Hunyuan FP8, I2V, and tiny debug runs.

The server exposes single-job and batch APIs:

```text
POST /v1/video/generations
GET  /v1/video/generations/<job_id>
GET  /v1/video/generations/<job_id>/logs
GET  /v1/video/generations/<job_id>/video
POST /v1/video/generation-batches
GET  /v1/video/generation-batches
GET  /v1/video/generation-batches/<batch_id>
```

## Documentation

- [Run the video backend](docs/video-backend-runbook.md): full setup, server, API, and WorldOdyssey runbook.
- [Tiny models and Diffusers](docs/tiny-models-and-diffusers.md): tiny Wan fixtures, Diffusers/FastVideo benchmarks,
  and backend slow tests.
- [SGLang Diffusion setup](references/sglang-diffusion.md): pinned SGLang environment, CUDA probes, and failure modes.
- [Video backend contract](references/video-backend.md): API schema, provider capability behavior, and runtime details.
- [Submission configs](configs/README.md): YAML config shape and `--set dotted.path=value` overrides.

## Environment

Use the repository `.venv` directly. Do not use `uv run`.

```bash
cd /home/coder/Python_project/WorldOdyssey_inference
source .venv/bin/activate
```

Install dependencies through the activated uv-managed environment:

```bash
source .venv/bin/activate
uv pip install <package>
```

## Setup

Run setup once from the repository root:

```bash
bash scripts/setup_video_backend.sh
```

This command:

- installs main backend dependencies into `.venv` with `uv sync --inexact`
- installs the unified SGLang Diffusion runtime into `.venv_sglang`
- keeps SGLang isolated from the main Diffusers/FastVideo environment
- verifies the main server packages after installation

Do not use conda for this setup unless explicitly approved. Do not use `uv run`.

## Start SGLang

Start the native SGLang Diffusion server first. For the FastWan T2V VSA path:

```bash
WORLDODYSSEY_SGLANG_NUM_GPUS=1 \
bash scripts/serve_sglang_diffusion.sh FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --attention-backend video_sparse_attn \
  --VSA-sparsity 0.5
```

For I2V, start a matching I2V workload and model:

```bash
WORLDODYSSEY_SGLANG_WORKLOAD_TYPE=i2v \
WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory \
WORLDODYSSEY_SGLANG_LOG_LEVEL=debug \
WORLDODYSSEY_SGLANG_NUM_GPUS=1 \
bash scripts/serve_sglang_diffusion.sh weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers
```

For Hunyuan FP8, use the same native server launcher and pass the FP8 transformer override:

```bash
WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory \
WORLDODYSSEY_SGLANG_LOG_LEVEL=debug \
WORLDODYSSEY_SGLANG_NUM_GPUS=2 \
WORLDODYSSEY_SGLANG_TP_SIZE=1 \
WORLDODYSSEY_SGLANG_SP_DEGREE=2 \
bash scripts/serve_sglang_diffusion.sh hunyuanvideo-community/HunyuanVideo \
  --transformer-path lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer
```

### Memory Offload

Use the launcher offload preset when a model starts or runs out of VRAM. This is a startup policy for the persistent
native SGLang server, not a request YAML field.

```bash
WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory \
WORLDODYSSEY_SGLANG_LOG_LEVEL=debug \
WORLDODYSSEY_SGLANG_NUM_GPUS=1 \
bash scripts/serve_sglang_diffusion.sh weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers
```

The `memory` preset expands to SGLang-Diffusion's low-memory flags:

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

`--dit-offload-prefetch-size 0` is the lowest-memory setting. If the server is stable but too slow, pass a later value
after the model id, for example `--dit-offload-prefetch-size 0.1`; trailing SGLang flags override the preset because
the launcher appends them after the preset args. The launcher defaults `SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1` for
this preset unless you set it yourself; this keeps the low-memory path on native norm fallbacks instead of fragile
CuTe DSL fused norm kernels. It also defaults `USE_TRITON_W8A8_FP8_KERNEL=1` so ModelOpt FP8 uses SGLang's Triton
W8A8 FP8 branch instead of the CUTLASS branch that failed on this RTX 4090 host, and defaults
`SGLANG_DISABLE_FLASHINFER_ROPE=1` so Wan InP uses SGLang's Triton RoPE fallback instead of FlashInfer's JIT path that
requires `nvcc`. It also defaults `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless you set a custom allocator
config. The VAE tile defaults can be overridden with `WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_MIN_HEIGHT`,
`WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_MIN_WIDTH`, `WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_STRIDE_HEIGHT`,
`WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_STRIDE_WIDTH`, `WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_MIN_NUM_FRAMES`, and
`WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_STRIDE_NUM_FRAMES`, or by passing trailing `--vae-config.*` flags. The launcher
rejects this preset when `SGLANG_CACHE_DIT_ENABLED=true`.

The launcher prints the backend URL, API format, and an optional model hint:

```bash
export WORLDODYSSEY_SGLANG_BASE_URL=http://127.0.0.1:30000
export WORLDODYSSEY_SGLANG_VIDEO_API_FORMAT=multipart
export WORLDODYSSEY_SGLANG_MODEL=<loaded-model-id>
```

Keep that server running. To stop it, press `Ctrl-C` in the foreground terminal, or send `Ctrl-C` to the tmux pane if
you started it in tmux.

## Start The Backend

In a second shell, point the provider-neutral backend at the SGLang server:

```bash
export WORLDODYSSEY_SGLANG_BASE_URL=http://127.0.0.1:30000
source .venv/bin/activate
python scripts/serve_video_backend.py --host 127.0.0.1 --port 8000
```

`WORLDODYSSEY_SGLANG_VIDEO_API_FORMAT` defaults to `multipart`, matching the unified native server launcher. Set it to
`json` only when intentionally using the legacy SGLang 0.5.5 wrapper stack.

`WORLDODYSSEY_SGLANG_MODEL` is optional metadata for provider discovery. The backend does not require it and does not
enforce that it matches each request; the `model` field is forwarded to SGLang's native `/v1/videos` endpoint.

In another shell, verify it before submitting jobs:

```bash
source .venv/bin/activate
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/video/providers | python -m json.tool
```

Expected health response:

```json
{"status":"ok"}
```

The `sglang` provider should be enabled and list both `text_to_video` and `image_to_video`. The provider capability
should report `server_api: /v1/videos` and the selected `server_api_format`.

For long-lived use, start the same command in tmux after a short foreground validation. Leave the tmux session open so
logs remain available.

## Submit FastWan T2V VSA

Submit a direct local SGLang FastWan VSA job. VSA is configured on the SGLang server launcher above, not in the request:

```bash
curl -X POST http://127.0.0.1:8000/v1/video/generations \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "sglang",
    "model": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    "mode": "text_to_video",
    "prompt": "A calm ocean wave at sunrise",
    "options": {
      "height": 448,
      "width": 832,
      "num_frames": 61,
      "timeout_seconds": 600
    }
  }'
```

Poll the returned job id:

```bash
curl http://127.0.0.1:8000/v1/video/generations/<job_id> | python -m json.tool
```

Read logs:

```bash
curl http://127.0.0.1:8000/v1/video/generations/<job_id>/logs
```

Download the result after the job reaches `succeeded`:

```bash
curl -L -o output.mp4 http://127.0.0.1:8000/v1/video/generations/<job_id>/video
```

## Submit Hunyuan FP8

Start SGLang with the Hunyuan FP8 command above, start the FastAPI backend normally, then submit through the same
provider-neutral JSON API:

```bash
curl -X POST http://127.0.0.1:8000/v1/video/generations \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "sglang",
    "model": "hunyuanvideo-community/HunyuanVideo",
    "mode": "text_to_video",
    "prompt": "A tiny red cube rotating on a plain gray background",
    "options": {
      "height": 128,
      "width": 128,
      "num_frames": 5,
      "num_inference_steps": 1,
      "seed": 1,
      "timeout_seconds": 600
    }
  }'
```

For this FP8 path, the SGLang server receives native multipart fields. The FP8 transformer id belongs on the server
launcher as `--transformer-path`; request `model` stays as `hunyuanvideo-community/HunyuanVideo`.

## Run WorldOdyssey

The imported example task lives at:

```text
compiled_resources/worldodyssey/WorldOdyssey/inputs/move_bookmark
```

You can also pass the parent inputs directory:

```text
compiled_resources/worldodyssey/WorldOdyssey/inputs
```

When the path contains child task folders, the submitter expands it into a `/v1/video/generation-batches` request.
Direct child folders without `task.json` are recorded in batch metadata as `skipped_entries`.

The WorldOdyssey adapter reads `task.json`, uses only the `task` field as the generation prompt, and keeps the topology
graph plus source frame/video paths in request metadata.

Dry-run the default T2V request:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --dry-run
```

Run the checked-in T2V YAML config:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-move-bookmark-t2v.yaml
```

Run the checked-in I2V YAML config:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-move-bookmark-i2v.yaml
```

Run the checked-in parent-input T2V batch config:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-inputs-batch-t2v.yaml
```

Run the low-memory tinywan variant for quick parent-input batch debugging:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-inputs-batch-t2v-tinywan.yaml
```

Run the Hunyuan FP8 parent-input smoke config after starting the FP8 server/backend pair:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-inputs-batch-t2v-hunyuan-fp8.yaml
```

Run a larger supported 540p FP8 config for visual inspection of one task. Start the Hunyuan FP8 server with
`WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory` first so the low-memory FP8 and VAE tile defaults are active:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-t2v-hunyuan-fp8-visual.yaml
```

That config writes:

```text
artifacts/video-backend/worldodyssey-move-bookmark-hunyuan-fp8-visual.mp4
```

The backend must already be running for the submit commands above.

Dry-run all available task folders under the imported `inputs/` root:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  compiled_resources/worldodyssey/WorldOdyssey/inputs \
  --dry-run
```

Submit that parent directory as a batch and download successful child outputs by task id:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-inputs-batch-t2v.yaml
```

For batch submissions, use `run.download_dir` or `--download-dir`. `run.download_path` is only for single-task
submissions.

## I2V Smoke Run

Before running I2V, restart native SGLang with the target I2V model. The FastAPI backend can stay up if
`WORLDODYSSEY_SGLANG_BASE_URL` still points at the same SGLang host and port.

Use a short I2V run before production settings:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-i2v.yaml \
  --set request.options.num_frames=17 \
  --set 'adapter.prompt_prefix=Generate an egocentric first-person video.' \
  --set run.download_path=artifacts/video-backend/worldodyssey-move-bookmark-i2v-smoke.mp4
```

The default WorldOdyssey prompt is only the task text. Use `adapter.prompt_prefix` or `--prompt-prefix` when an
inference run needs extra instruction text prepended before the task.

Example dry-run with a prefix:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-i2v.yaml \
  --set 'adapter.prompt_prefix=Generate an egocentric first-person video.' \
  --dry-run
```

Example I2V run with a custom local image path instead of the WorldOdyssey main frame:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-i2v.yaml \
  --set adapter.include_main_image_base64=false \
  --set request.image_path=/absolute/path/to/input.png \
  --set request.options.num_frames=17 \
  --set run.download_path=artifacts/video-backend/custom-image-i2v-smoke.mp4
```

The local provider accepts exactly one image input. When using `request.image_path`, keep
`adapter.include_main_image_base64=false` so the adapter does not also attach `frames/main.png` as `image_base64`.

## YAML Overrides

Override any config value from the command line:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-i2v.yaml \
  --set request.options.num_frames=17 \
  --set run.download_path=artifacts/video-backend/test-i2v.mp4
```

Override precedence is:

```text
built-in defaults < YAML config < named CLI flags < --set dotted.path=value
```

## Outputs

Generated backend outputs are written under `artifacts/video-backend/`:

- `jobs/<job_id>.json`: durable job state
- `batches/<batch_id>.json`: durable batch state and associated job ids
- `logs/<job_id>.log`: provider stdout/stderr
- `videos/<job_id>/output.mp4`: generated video

WorldOdyssey submitter configs can also download a single-task copy to `run.download_path` or batch copies to
`run.download_dir`.

## Native SGLang Debugging

The backend does not run `sglang generate` and does not run a Python one-shot wrapper. It posts to a persistent native
SGLang server at `WORLDODYSSEY_SGLANG_BASE_URL`:

```text
POST /v1/videos
GET  /v1/videos/<sglang_video_id>
GET  /v1/videos/<sglang_video_id>/content
```

The unified native multipart server path accepts `prompt`, `model`, `size`, `fps`, `num_frames`, optional `seconds`,
optional I2V `input_reference`, `negative_prompt`, `num_inference_steps`, `seed`, `guidance_scale`, and
`options.provider_options.request_fields`. Launch-time settings such as GPU count, VSA, tensor parallelism, sequence
parallelism, FP8 transformer overrides, and memory offload still belong on `scripts/serve_sglang_diffusion.sh`.

For direct stack debugging, activate the isolated environment and use the same runtime guards:

```bash
source .venv_sglang/bin/activate
export PATH="$PWD/.venv_sglang/bin:/usr/local/bin:/usr/bin:/bin"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME="$PWD/.venv_sglang/lib/python3.12/site-packages/nvidia"
sglang serve --help
```

Use `sglang serve --help` as the installation smoke test for the diffusion server path. Job logs created by this
backend should start with `SGLang server:` and `POST /v1/videos`.

The unified SGLang stack uses `sglang-kernel==0.4.1` with CUDA 12.8 Torch wheels and should still avoid CUDA 13 runtime
packages. See `references/sglang-diffusion.md` for the full installation and validation notes.

## Validated Status

Current validation status:

- provider unit tests cover the native `/v1/videos` create, poll, and download flow using a fake SGLang server
- provider unit tests cover multipart FP8 request fields and I2V file upload
- WorldOdyssey parent `inputs/` dry-run expansion into a batch request is covered by tests
- YAML dry-run and override behavior for T2V/I2V is covered by tests
- real FastWan T2V VSA server startup was validated with `WORLDODYSSEY_SGLANG_NUM_GPUS=2` until Uvicorn served
  `http://127.0.0.1:30000`
- the README FastWan T2V curl was validated end to end through the API on 2026-05-17; it completed in 17.0533 seconds
  and wrote `artifacts/video-backend/videos/vid_20260517T185746Z_3497cb08/output.mp4`
- native Hunyuan FP8 SGLang multipart smoke was validated on 2026-05-18 and wrote
  `artifacts/video-backend/75a30c04-59f1-4255-ba8c-6ac9aee91d1b.mp4`
- Hunyuan FP8 was validated through the provider-neutral FastAPI app on 2026-05-18; job
  `vid_20260518T065703Z_04fe6059` wrote
  `artifacts/video-backend-api-smoke/videos/vid_20260518T065703Z_04fe6059/output.mp4`
- the larger WorldOdyssey Hunyuan FP8 visual config was validated end to end on 2026-05-18 with the `memory` preset;
  job `vid_20260518T113402Z_9c852dfc` wrote
  `artifacts/video-backend/worldodyssey-move-bookmark-hunyuan-fp8-visual.mp4`
- full real I2V inference through the API remains a slow GPU validation step

## Tiny Wan Batch Debugging

Use the tiny Wan batch config for low-memory API debugging. The batch endpoint creates normal generation jobs and
tracks them under one `batch_id`; with the default `--job-workers 1`, those jobs run sequentially to avoid local GPU
OOM.

Dry-run the checked-in batch payload:

```bash
source .venv/bin/activate
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml --dry-run
```

Submit it and wait for completion:

```bash
source .venv/bin/activate
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml
```

Override any item with list indexes:

```bash
source .venv/bin/activate
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml \
  --set 'requests.0.prompt=A tiny camera circles a glass cube.' \
  --set requests.1.options.num_frames=5
```

Direct API shape:

```bash
curl -X POST http://127.0.0.1:8000/v1/video/generation-batches \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "provider": "sglang",
        "model": "Erland/tiny-wan2.1-t2v-debug",
        "mode": "text_to_video",
        "prompt": "A camera glides over a quiet city street at night.",
        "options": {
          "height": 64,
          "width": 64,
          "num_frames": 5,
          "timeout_seconds": 300
        }
      }
    ],
    "metadata": {
      "purpose": "tiny-wan-batch-debug"
    }
  }'
```

Poll the returned batch id:

```bash
curl http://127.0.0.1:8000/v1/video/generation-batches/<batch_id> | python -m json.tool
```
