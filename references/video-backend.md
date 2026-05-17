# Video Backend Server

The video backend server exposes one provider-neutral API for video generation while keeping provider-specific details
behind adapters. The first enabled adapter is local SGLang Diffusion.

The local SGLang adapter is native-server-only:

- start SGLang with `scripts/serve_sglang_diffusion.sh`
- start the FastAPI backend with `WORLDODYSSEY_SGLANG_BASE_URL`
- submit jobs through `/v1/video/generations` or `/v1/video/generation-batches`
- the backend posts to SGLang's native `/v1/videos` API and downloads `/v1/videos/<id>/content`

There is no one-shot generation fallback, no `sglang generate` backend path, and no Python single-request runner.

## Setup

Run the setup command from the repository root:

```bash
bash scripts/setup_video_backend.sh
```

This command:

- syncs the main `.venv` dependencies for the FastAPI server with `uv sync --inexact`
- installs or updates the isolated `.venv_sglangcuda12` SGLang Diffusion environment
- verifies the server packages are importable

It does not start SGLang or the FastAPI backend.

## Runtime Topology

Start native SGLang first:

```bash
WORLDODYSSEY_SGLANG_WORKLOAD_TYPE=t2v \
WORLDODYSSEY_SGLANG_NUM_GPUS=1 \
bash scripts/serve_sglang_diffusion.sh FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --attention-backend video_sparse_attn \
  --VSA-sparsity 0.5
```

Then start the backend in another shell:

```bash
export WORLDODYSSEY_SGLANG_BASE_URL=http://127.0.0.1:30000
source .venv/bin/activate
python scripts/serve_video_backend.py --host 127.0.0.1 --port 8000
```

`WORLDODYSSEY_SGLANG_MODEL` is optional metadata for provider discovery. The backend forwards `request.model` to
SGLang's native `/v1/videos` endpoint and does not validate it against an environment variable.

## API

Health:

```bash
curl http://127.0.0.1:8000/health
```

Provider capabilities:

```bash
curl http://127.0.0.1:8000/v1/video/providers
```

The `sglang` capability exposes:

- `modes`: `text_to_video`, `image_to_video`
- `models`: empty list, because the backend does not maintain a model allowlist
- `setup.server_script`: `scripts/serve_sglang_diffusion.sh`
- `setup.server_api`: `/v1/videos`
- `setup.configured_server_model_hint`: optional value from `WORLDODYSSEY_SGLANG_MODEL`

Remote providers are visible but disabled until adapters are implemented:

- `fal`: fal.ai video providers and `FAL_KEY`
- `google_veo`: Google Veo operation polling and `GEMINI_API_KEY`
- `xai_grok`: Grok Imagine request polling and `XAI_API_KEY`

## Single Job

Submit a local SGLang T2V job:

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

Submit a local SGLang I2V job:

```bash
curl -X POST http://127.0.0.1:8000/v1/video/generations \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "sglang",
    "mode": "image_to_video",
    "model": "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
    "prompt": "Generate a first-person video of a hand moving the bookmark into the yellow book.",
    "image_path": "compiled_resources/worldodyssey/WorldOdyssey/inputs/move_bookmark/frames/main.png",
    "options": {
      "height": 256,
      "width": 448,
      "num_frames": 17,
      "timeout_seconds": 900
    }
  }'
```

Poll:

```bash
curl http://127.0.0.1:8000/v1/video/generations/<job_id>
```

Fetch logs:

```bash
curl http://127.0.0.1:8000/v1/video/generations/<job_id>/logs
```

Download the video after the job reaches `succeeded`:

```bash
curl -L -o output.mp4 http://127.0.0.1:8000/v1/video/generations/<job_id>/video
```

## Batch Job

Batch submission creates one child job per request and returns a `batch_id` that tracks aggregate status:

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

Poll the batch:

```bash
curl http://127.0.0.1:8000/v1/video/generation-batches/<batch_id>
```

Each child job keeps its own logs and video endpoint.

## Request Schema

Canonical request fields:

```json
{
  "provider": "sglang",
  "model": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
  "mode": "text_to_video",
  "prompt": "A calm ocean wave at sunrise",
  "negative_prompt": null,
  "image_path": null,
  "image_url": null,
  "image_base64": null,
  "end_image_url": null,
  "reference_image_urls": [],
  "video_url": null,
  "options": {
    "duration": null,
    "num_frames": 61,
    "resolution": null,
    "width": 832,
    "height": 448,
    "aspect_ratio": null,
    "fps": null,
    "seed": null,
    "guidance_scale": null,
    "num_inference_steps": null,
    "num_gpus": 1,
    "generate_audio": null,
    "attention_backend": null,
    "vsa_sparsity": null,
    "timeout_seconds": 300,
    "log_level": "info",
    "provider_options": {}
  },
  "metadata": {}
}
```

The shared schema intentionally contains fields needed by future providers. For local `provider=sglang`, native
`/v1/videos` only accepts a smaller request surface. The provider rejects:

- `options.num_inference_steps`
- `options.seed`
- `options.guidance_scale`
- `options.num_gpus` when not the default `1`
- `options.attention_backend`
- `options.vsa_sparsity`
- `options.provider_options`

Configure model id, workload type, GPU count, VSA, and server-side sampling behavior on `scripts/serve_sglang_diffusion.sh`.
If native SGLang does not expose a setting on the diffusion server entrypoint, the backend does not invent an alternate
execution path.

## Local SGLang Behavior

The local SGLang adapter validates request shape and server identity:

- Supported modes: `text_to_video`, `image_to_video`
- T2V requires explicit `height`, `width`, and `num_frames`
- I2V accepts exactly one of `image_path`, `image_url`, or `image_base64`
- I2V stages `image_url` and `image_base64` into the job folder before calling SGLang
- The provider sends `size` as `<width>x<height>` and `input_reference` for I2V
- Reference-image, end-image, edit, extension, and video-input requests fail fast with `unsupported_request`

The provider does not maintain a hardcoded SGLang model allowlist. Custom model ids are allowed as long as they match
the model loaded by the native SGLang server.

## WorldOdyssey Adapter

`scripts/submit_worldodyssey_task.py` adapts a WorldOdyssey task folder or `task.json` into the canonical
`POST /v1/video/generations` payload. If the provided path is a parent directory containing child task folders, it
adapts those children into `POST /v1/video/generation-batches`.

The default task path is:

```text
compiled_resources/worldodyssey/WorldOdyssey/inputs/move_bookmark
```

The parent input path is:

```text
compiled_resources/worldodyssey/WorldOdyssey/inputs
```

The generated prompt is only the WorldOdyssey `task` field by default. Use `adapter.prompt_prefix` or `--prompt-prefix`
to prepend additional instruction text.

Examples:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --dry-run

source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-move-bookmark-t2v.yaml

source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-inputs-batch-t2v.yaml
```

For parent-directory batches, use `run.download_dir` or `--download-dir`; `run.download_path` is only valid for
single-task submissions.

## Outputs

Generated files are stored under `artifacts/video-backend/`:

- `jobs/<job_id>.json`: durable job state
- `batches/<batch_id>.json`: durable batch state and associated job ids
- `logs/<job_id>.log`: provider log
- `videos/<job_id>/output.mp4`: generated video

The WorldOdyssey submitter can additionally download a single-task copy to `run.download_path` or batch child copies to
`run.download_dir`.

## Troubleshooting Signals

A current local SGLang job log starts with:

```text
SGLang server: http://127.0.0.1:30000
POST /v1/videos
```

The local launcher is `scripts/serve_sglang_diffusion.sh`. It calls `scripts/sglang_diffusion_serve.py`, which launches
SGLang's native diffusion server and applies the SGLang 0.5.x `/v1/videos` output-filename compatibility patch before
startup. There is still no one-shot generation fallback in the backend.

If `request.model` is incompatible with the running native SGLang server, the backend records SGLang's native
`/v1/videos` response in the job log.
