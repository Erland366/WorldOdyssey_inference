# Run the Video Backend

This runbook starts the provider-neutral video backend and submits inference jobs to a persistent native SGLang
Diffusion server. The local provider id is `sglang`.

There is no one-shot generation fallback. The backend never launches `sglang generate` and never launches a Python
single-request runner. It requires the native SGLang Diffusion server to already be running and calls SGLang's native
video API:

```text
POST /v1/videos
GET  /v1/videos/<sglang_video_id>
GET  /v1/videos/<sglang_video_id>/content
```

The main supported local T2V model is:

```text
FastVideo/FastWan2.1-T2V-1.3B-Diffusers
```

The common local I2V smoke model used by the checked-in config is:

```text
weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers
```

The validated T2V VSA launch settings are:

```text
--attention-backend video_sparse_attn
--VSA-sparsity 0.5
```

The validated FP8 T2V pair is:

```text
base model:       hunyuanvideo-community/HunyuanVideo
FP8 transformer:  lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer
```

## 1. Setup

Run from the repository root:

```bash
cd /home/coder/Python_project/WorldOdyssey_inference
bash scripts/setup_video_backend.sh
```

What this does:

- initializes the WorldOdyssey input repository submodule under `submodule/worldodyssey`
- installs main server dependencies into `.venv`
- installs the unified SGLang Diffusion runtime into `.venv_sglang`
- preserves existing local ML packages by using `uv sync --inexact`
- verifies key backend packages after installation

Do not use conda for this setup unless explicitly approved. Do not use `uv run`.

If you skip setup and only need the WorldOdyssey task inputs, initialize the submodule directly:

```bash
git submodule update --init --recursive submodule/worldodyssey
```

## 2. Start Native SGLang

Start SGLang in the foreground first so startup errors surface immediately:

```bash
WORLDODYSSEY_SGLANG_NUM_GPUS=1 \
bash scripts/serve_sglang_diffusion.sh FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --attention-backend video_sparse_attn \
  --VSA-sparsity 0.5
```

For I2V:

```bash
WORLDODYSSEY_SGLANG_WORKLOAD_TYPE=i2v \
WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory \
WORLDODYSSEY_SGLANG_LOG_LEVEL=debug \
WORLDODYSSEY_SGLANG_NUM_GPUS=1 \
bash scripts/serve_sglang_diffusion.sh weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers
```

For Hunyuan FP8:

```bash
WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory \
WORLDODYSSEY_SGLANG_LOG_LEVEL=debug \
WORLDODYSSEY_SGLANG_NUM_GPUS=2 \
WORLDODYSSEY_SGLANG_TP_SIZE=1 \
WORLDODYSSEY_SGLANG_SP_DEGREE=2 \
bash scripts/serve_sglang_diffusion.sh hunyuanvideo-community/HunyuanVideo \
  --transformer-path lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer
```

If the model OOMs during startup or inference, restart SGLang with the explicit memory offload preset:

```bash
WORLDODYSSEY_SGLANG_OFFLOAD_PRESET=memory \
WORLDODYSSEY_SGLANG_LOG_LEVEL=debug \
WORLDODYSSEY_SGLANG_NUM_GPUS=1 \
bash scripts/serve_sglang_diffusion.sh weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers
```

The preset is native-server-only and expands to `--dit-layerwise-offload`, `--dit-cpu-offload false`,
`--dit-offload-prefetch-size 0`, `--text-encoder-cpu-offload`, `--image-encoder-cpu-offload`, `--vae-cpu-offload`,
`--pin-cpu-memory`, `--vae-tiling`, `--vae-slicing`, and smaller VAE decode tiles:
`--vae-config.tile-sample-min-height 128`, `--vae-config.tile-sample-min-width 128`,
`--vae-config.tile-sample-stride-height 96`, `--vae-config.tile-sample-stride-width 96`,
`--vae-config.tile-sample-min-num-frames 8`, and `--vae-config.tile-sample-stride-num-frames 4`. It also defaults
`SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1`, `USE_TRITON_W8A8_FP8_KERNEL=1`,
`SGLANG_DISABLE_FLASHINFER_ROPE=1`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless those env vars are
already set. `SGLANG_DISABLE_FLASHINFER_ROPE=1` avoids FlashInfer's RoPE JIT path for Wan InP when the uv environment
does not provide `nvcc`; SGLang then uses its Triton RoPE fallback. Keep this as a launch-time setting; request YAML
should not carry offload knobs. Pass a later `--dit-offload-prefetch-size`, such as `0.1`, after the model id only
after the lowest-memory launch succeeds.

The launcher prints the backend URL, API format, and an optional model hint for that SGLang process:

```bash
export WORLDODYSSEY_SGLANG_BASE_URL=http://127.0.0.1:30000
export WORLDODYSSEY_SGLANG_VIDEO_API_FORMAT=multipart
export WORLDODYSSEY_SGLANG_MODEL=<loaded-model-id>
```

The backend forwards each request's `model` field to native SGLang and does not enforce it against
`WORLDODYSSEY_SGLANG_MODEL`. If you restart SGLang with another model on the same host and port, the FastAPI backend can
stay running.

For a long-lived SGLang server, follow the repository two-phase rule: validate in the foreground first, then start the
same command in tmux. Create the session from the repository root:

```bash
tmux new-session -s "tmux-$(( $(tmux list-sessions -F '#S' 2>/dev/null | sed -n 's/^Codex-\([0-9]\+\)$/\1/p' | sort -n | tail -n1 || echo 0) + 1 ))" -n "$(basename "$PWD")-1" -c "$PWD" -d
```

Then send the SGLang command to the tmux session:

```bash
tmux send-keys -t <session> 'WORLDODYSSEY_SGLANG_NUM_GPUS=1 bash scripts/serve_sglang_diffusion.sh FastVideo/FastWan2.1-T2V-1.3B-Diffusers --attention-backend video_sparse_attn --VSA-sparsity 0.5' Enter
```

Check the pane:

```bash
tmux capture-pane -t <session> -p -S -50
```

Leave the tmux session open after use so logs stay available.

## 3. Start The Backend

In a second shell, point the backend at the SGLang server:

```bash
cd /home/coder/Python_project/WorldOdyssey_inference
export WORLDODYSSEY_SGLANG_BASE_URL=http://127.0.0.1:30000
source .venv/bin/activate
python scripts/serve_video_backend.py --host 127.0.0.1 --port 8000
```

`WORLDODYSSEY_SGLANG_VIDEO_API_FORMAT` defaults to `multipart`, matching the unified native server launcher. Set it to
`json` only when intentionally using the legacy SGLang 0.5.5 wrapper stack.

`WORLDODYSSEY_SGLANG_MODEL` is optional provider metadata:

```bash
export WORLDODYSSEY_SGLANG_MODEL=weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers
```

Check health:

```bash
source .venv/bin/activate
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Inspect providers:

```bash
source .venv/bin/activate
curl http://127.0.0.1:8000/v1/video/providers | python -m json.tool
```

Current behavior:

- `sglang` is enabled and local.
- `sglang` accepts `text_to_video` and `image_to_video`.
- `sglang.setup.server_script` is `scripts/serve_sglang_diffusion.sh`.
- `sglang.setup.server_api` is `/v1/videos`.
- `sglang.setup.server_api_format` is `json` or `multipart`.
- `fal`, `google_veo`, and `xai_grok` are visible but disabled until adapters and API-key handling are implemented.

## 4. Submit T2V

VSA and GPU settings are SGLang server-launch settings. Do not put them in the generation request.

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

Poll:

```bash
curl http://127.0.0.1:8000/v1/video/generations/<job_id> | python -m json.tool
```

Read logs:

```bash
curl http://127.0.0.1:8000/v1/video/generations/<job_id>/logs
```

Download:

```bash
curl -L -o output.mp4 http://127.0.0.1:8000/v1/video/generations/<job_id>/video
```

## 4A. Submit Hunyuan FP8 T2V

The FastAPI request remains JSON. The local provider converts it to multipart for native SGLang.

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

The FP8 transformer repository is not the request model id. It is provided to SGLang at launch with
`--transformer-path lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer`.

Server-side outputs are stored under:

```text
artifacts/video-backend/jobs/<job_id>.json
artifacts/video-backend/logs/<job_id>.log
artifacts/video-backend/videos/<job_id>/output.mp4
```

## 5. Submit I2V

Start SGLang with the target I2V model before using this endpoint.

The easy I2V path needs an image input and prompt. If `mode` and `model` are omitted, the API infers
`mode=image_to_video` and the default local I2V model from the image field. For a custom model, set `model` explicitly:

```bash
curl -X POST http://127.0.0.1:8000/v1/video/generations \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "sglang",
    "mode": "image_to_video",
    "model": "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
    "prompt": "Generate a first-person video of a hand moving the bookmark into the yellow book.",
    "image_path": "submodule/worldodyssey/inputs/move_bookmark/frames/main.png",
    "options": {
      "height": 256,
      "width": 448,
      "num_frames": 17,
      "timeout_seconds": 900
    }
  }'
```

Supported local image inputs:

- `image_path`: local path visible to the backend process
- `image_url`: HTTP/HTTPS URL that the backend downloads into the job folder
- `image_base64`: raw base64 or a `data:image/...;base64,...` URI

Use exactly one image input per local SGLang I2V request.

## 6. Request Fields

The unified native multipart server forwards these request-time fields to SGLang:

- `options.seed`
- `options.guidance_scale`
- `negative_prompt`
- `options.num_inference_steps`
- scalar `options.provider_options.request_fields`
- structured `options.provider_options.extra_body` as a JSON string form field

The local provider rejects launch-time settings in generation requests: non-default `options.num_gpus`,
`options.attention_backend`, and `options.vsa_sparsity`. Configure model id, GPU count, VSA, tensor parallelism,
sequence parallelism, transformer overrides, and memory offload on `scripts/serve_sglang_diffusion.sh`. If SGLang does
not expose a setting on its native diffusion server, this backend does not invent a fallback path.

## 7. Submit a Tiny Wan Batch

The tiny Wan batch config is the low-memory path for testing batch submission shape:

```text
configs/tiny-wan-batch.yaml
```

Dry-run the YAML:

```bash
source .venv/bin/activate
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml --dry-run
```

Submit and wait:

```bash
source .venv/bin/activate
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml
```

Override one item without editing the file:

```bash
source .venv/bin/activate
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml \
  --set 'requests.0.prompt=A tiny camera circles a glass cube.' \
  --set requests.1.options.num_frames=5
```

Equivalent direct API shape:

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
      },
      {
        "provider": "sglang",
        "model": "Erland/tiny-wan2.1-t2v-debug",
        "mode": "text_to_video",
        "prompt": "A tiny toy boat floats across a still pond.",
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

Each child job still has its own log and video endpoint:

```bash
curl http://127.0.0.1:8000/v1/video/generations/<job_id>/logs
curl -L -o output.mp4 http://127.0.0.1:8000/v1/video/generations/<job_id>/video
```

## 8. Run WorldOdyssey Inputs

WorldOdyssey inputs are tracked as a git submodule at `submodule/worldodyssey`. The setup script initializes it; if the
task paths below are missing, run:

```bash
git submodule update --init --recursive submodule/worldodyssey
```

The imported task lives at:

```text
submodule/worldodyssey/inputs/move_bookmark
```

The parent input root is:

```text
submodule/worldodyssey/inputs
```

The adapter reads `task.json`, uses only the `task` field as the generation prompt, and keeps the topology graph plus
source frame/video paths in request metadata. Use `adapter.prompt_prefix` only when the run needs extra instruction text
prepended before the task.

Dry-run the default payload:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --dry-run
```

Run the checked-in T2V config:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-move-bookmark-t2v.yaml
```

Run the checked-in I2V config:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-move-bookmark-i2v.yaml
```

Run the parent `inputs/` directory as a batch:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-inputs-batch-t2v.yaml
```

Run the low-memory I2V parent-input smoke config:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-inputs-batch-t2v-wan-InP.yaml
```

Run the Hunyuan FP8 parent-input smoke config:

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

For parent-directory batches, use `run.download_dir` or `--download-dir`. Single-task `run.download_path` is rejected
for batch submissions because there is no single output filename.

## 9. Custom I2V Image Path

The WorldOdyssey I2V adapter attaches the task's main frame as `image_base64` by default. To use a custom image path,
disable that attachment and provide `request.image_path`:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-i2v.yaml \
  --set adapter.include_main_image_base64=false \
  --set request.image_path=/absolute/path/to/input.png \
  --set request.options.num_frames=17 \
  --set run.download_path=artifacts/video-backend/custom-image-i2v-smoke.mp4
```

## 10. Current Validation Status

Current automated validation covers:

- provider payload mapping to native `/v1/videos`
- provider create, poll, and download flow against a fake SGLang server
- strict rejection of one-shot-only SGLang request fields
- multipart FP8 request forwarding for seed, step count, negative prompt, provider fields, and I2V file upload
- WorldOdyssey prompt construction and parent-input batch expansion
- YAML dry-run and `--set dotted.path=value` override behavior
- FastWan T2V VSA server startup with `WORLDODYSSEY_SGLANG_NUM_GPUS=2` until Uvicorn served
  `http://127.0.0.1:30000`
- the FastWan T2V curl in this runbook was validated end to end through the API on 2026-05-17; it completed in
  17.0533 seconds and wrote `artifacts/video-backend/videos/vid_20260517T185746Z_3497cb08/output.mp4`
- native Hunyuan FP8 SGLang multipart smoke was validated on 2026-05-18 and wrote
  `artifacts/video-backend/75a30c04-59f1-4255-ba8c-6ac9aee91d1b.mp4`
- Hunyuan FP8 was validated through the provider-neutral FastAPI app on 2026-05-18; job
  `vid_20260518T065703Z_04fe6059` wrote
  `artifacts/video-backend-api-smoke/videos/vid_20260518T065703Z_04fe6059/output.mp4`
- the larger WorldOdyssey Hunyuan FP8 visual config was validated end to end on 2026-05-18 with the `memory` preset;
  job `vid_20260518T113402Z_9c852dfc` wrote
  `artifacts/video-backend/worldodyssey-move-bookmark-hunyuan-fp8-visual.mp4`

Full real I2V inference through the API is a slow GPU validation step and was not rerun in this change.

## 11. Troubleshooting

If SGLang cannot find CUDA runtime libraries, rerun:

```bash
bash scripts/setup_video_backend.sh
```

The SGLang launcher sets these runtime guards:

```text
PATH=<repo>/.venv_sglang/bin:/usr/local/bin:/usr/bin:/bin
CC=/usr/bin/gcc
CXX=/usr/bin/g++
CUDA_HOME=<repo>/.venv_sglang/lib/python3.12/site-packages/nvidia
```

If a job fails, inspect the job record and log:

```bash
curl http://127.0.0.1:8000/v1/video/generations/<job_id> | python -m json.tool
curl http://127.0.0.1:8000/v1/video/generations/<job_id>/logs
```

The first log lines from this backend should be:

```text
SGLang server: http://127.0.0.1:30000
POST /v1/videos
```

The backend does not validate `request.model` against an environment variable. If a request uses a model incompatible
with the running SGLang server, inspect the job log for SGLang's native `/v1/videos` response.

If the server is not responding, check tmux:

```bash
tmux list-sessions
tmux capture-pane -t <session> -p -S -80
```
