# Video Backend Server

The video backend server exposes one local API for video generation while keeping provider-specific details behind
adapters. The first enabled adapter is local SGLang Diffusion with
`FastVideo/FastWan2.1-T2V-1.3B-Diffusers` and `video_sparse_attn`.

The API is intentionally job-based. Video generation is long-running for local SGLang and for remote services such as
fal.ai, Google Veo, and xAI Grok Imagine, so clients submit a job, poll for status, then download the result.

## Setup

Run the setup command from the repository root:

```bash
bash scripts/setup_video_backend.sh
```

This command:

- syncs the main `.venv` dependencies for the FastAPI server with `uv sync --inexact`, so existing local ML packages
  that are outside `pyproject.toml` stay installed
- installs or updates the isolated `.venv_sglangcuda12` SGLang Diffusion environment
- verifies the server packages are importable

It does not start the server.

## Start

Run the server in the foreground:

```bash
source .venv/bin/activate
python scripts/serve_video_backend.py --host 127.0.0.1 --port 8000
```

For a long-lived local server, use the repository tmux execution contract after a short foreground validation.

## API

Health:

```bash
curl http://127.0.0.1:8000/health
```

Provider capabilities:

```bash
curl http://127.0.0.1:8000/v1/video/providers
```

Submit a local SGLang FastWan VSA job:

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
      "num_inference_steps": 3,
      "seed": 123,
      "attention_backend": "video_sparse_attn",
      "vsa_sparsity": 0.5
    }
  }'
```

The response contains an `id`. Poll it:

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

Generated server outputs live under `artifacts/video-backend/`:

- `jobs/<job_id>.json`: durable job state
- `logs/<job_id>.log`: provider stdout/stderr
- `videos/<job_id>/output.mp4`: generated video

Validated local server run on May 16, 2026:

- Job id: `vid_20260516T135544Z_503ccabb`
- Status: `succeeded`
- Output: `artifacts/video-backend/videos/vid_20260516T135544Z_503ccabb/output.mp4`
- File size: `854463` bytes
- `/v1/video/generations/vid_20260516T135544Z_503ccabb/video` returned `200 video/mp4`
- Backend elapsed time: `82.1589` seconds

## Request Contract

The request body is provider-neutral:

```json
{
  "provider": "sglang",
  "model": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
  "mode": "text_to_video",
  "prompt": "A calm ocean wave at sunrise",
  "negative_prompt": null,
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
    "seed": 123,
    "guidance_scale": null,
    "num_inference_steps": 3,
    "num_gpus": 1,
    "generate_audio": null,
    "attention_backend": "video_sparse_attn",
    "vsa_sparsity": 0.5,
    "timeout_seconds": 300,
    "log_level": "info",
    "provider_options": {}
  },
  "metadata": {}
}
```

The local SGLang adapter is deliberately strict:

- Supported mode: `text_to_video`
- Supported model: `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`
- Explicitly set `num_frames`, `height`, and `width`; the adapter rejects local requests that omit them
- Explicitly set `attention_backend=video_sparse_attn` and `vsa_sparsity=0.5` for the validated VSA path
- Image, reference-image, edit, and extension requests currently fail fast with `unsupported_request`

Remote providers are listed by `GET /v1/video/providers` but are disabled until their adapters are implemented:

- `fal`: Seedance image-to-video shape and `FAL_KEY`
- `google_veo`: Veo operation-polling shape and `GEMINI_API_KEY`
- `xai_grok`: Grok Imagine request-id polling shape and `XAI_API_KEY`

Keeping disabled providers visible makes client-side capability discovery stable without pretending unsupported routes
work.

## Response Contract

Submit response:

```json
{
  "id": "vid_20260516T000000Z_1234abcd",
  "status": "queued",
  "provider": "sglang",
  "model": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
  "mode": "text_to_video",
  "created_at": "2026-05-16T00:00:00+00:00",
  "updated_at": "2026-05-16T00:00:00+00:00",
  "output": null,
  "metrics": {},
  "error": null
}
```

Successful poll response includes:

```json
{
  "status": "succeeded",
  "output": {
    "video_url": "/v1/video/generations/<job_id>/video",
    "local_path": "artifacts/video-backend/videos/<job_id>/output.mp4",
    "content_type": "video/mp4",
    "file_size": 854463
  },
  "metrics": {
    "elapsed_seconds": 21.39,
    "returncode": 0
  }
}
```

Failed jobs return `status: "failed"` and an `error` object with `code`, `message`, optional `provider_code`, and
`retryable`.

## Runtime Details

The local SGLang provider always injects these runtime guards into its subprocess:

```text
PATH=<repo>/.venv_sglangcuda12/bin:/usr/local/bin:/usr/bin:/bin
CC=/usr/bin/gcc
CXX=/usr/bin/g++
CUDA_HOME=<repo>/.venv_sglangcuda12/lib/python3.12/site-packages/nvidia
```

These are required on this host to avoid Miniconda CUDA/compiler tools. See `references/sglang-diffusion.md` for the
standalone SGLang validation path and failure modes.
