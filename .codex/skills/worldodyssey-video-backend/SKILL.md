---
name: worldodyssey-video-backend
description: Use and extend the WorldOdyssey provider-neutral video backend for local SGLang T2V/I2V inference, YAML submission configs, and future remote video providers.
---

# WorldOdyssey Video Backend

## Purpose

Use this skill when submitting WorldOdyssey inference jobs through the local provider-neutral video backend, debugging
the SGLang video API path, adding YAML submission configs, or extending the backend toward fal.ai, Google Veo, or xAI
Grok adapters.

## Context

The backend exposes job-based APIs:

- `POST /v1/video/generations`
- `GET /v1/video/generations/<job_id>`
- `GET /v1/video/generations/<job_id>/logs`
- `GET /v1/video/generations/<job_id>/video`
- `POST /v1/video/generation-batches`
- `GET /v1/video/generation-batches/<batch_id>`

The enabled local provider is `sglang`. Remote providers are visible as disabled capabilities until their adapters are
implemented.

## Recommended Practice

Start native SGLang first:

```bash
WORLDODYSSEY_SGLANG_WORKLOAD_TYPE=t2v \
WORLDODYSSEY_SGLANG_NUM_GPUS=1 \
bash scripts/serve_sglang_diffusion.sh FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
  --attention-backend video_sparse_attn \
  --VSA-sparsity 0.5
```

Then start the provider-neutral backend from the main environment:

```bash
export WORLDODYSSEY_SGLANG_BASE_URL=http://127.0.0.1:30000
source .venv/bin/activate
python scripts/serve_video_backend.py --host 127.0.0.1 --port 8000
```

`WORLDODYSSEY_SGLANG_MODEL` is optional metadata. The backend forwards each request's `model` to native SGLang and
does not reject requests based on a backend-side model hint.

For repeated WorldOdyssey inference, prefer YAML configs:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-move-bookmark-i2v.yaml
```

Inspect payloads before long runs:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-i2v.yaml \
  --dry-run
```

Override config values with dotted paths:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-i2v.yaml \
  --set request.options.num_frames=17 \
  --set adapter.prompt_prefix="Generate an egocentric first-person video." \
  --set run.download_path=artifacts/video-backend/test-i2v.mp4
```

By default, the adapter sends only the WorldOdyssey `task` field as the generation prompt. Use
`adapter.prompt_prefix` or `--prompt-prefix` when an inference run needs extra instruction text prepended to that task.

For low-memory batch debugging, use the tiny Wan batch config:

```bash
source .venv/bin/activate
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml --dry-run
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml
```

Batch `--set` overrides support list indexes, for example `requests.0.prompt=...`.

For a WorldOdyssey parent input directory, pass the parent path to `scripts/submit_worldodyssey_task.py`. The submitter
scans direct child folders with `task.json` and submits a batch:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py compiled_resources/worldodyssey/WorldOdyssey/inputs --dry-run
```

Use `--download-dir`, not `--download-path`, for parent-directory batch outputs.

## Local Model Defaults

T2V default:

- Model: `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`
- Shape: `448x832`
- Frames: `61`
- Attention and VSA: launch-time SGLang server settings, not request fields

T2V tiny debug:

- Model: `Erland/tiny-wan2.1-t2v-debug`
- Shape: `64x64`
- Frames: `5`
- Attention: omit from request YAML
- Runtime: native SGLang Diffusion server; no backend staging or one-shot fallback

I2V default:

- Model: `FastVideo/FastWan2.2-TI2V-5B-Diffusers`
- Shape: `480x832`
- Frames: `81`
- Attention: omit by default
- GPUs: launch-time SGLang server setting

## Failure Modes

If image input is rejected, confirm the request mode is `image_to_video` and exactly one of `image_path`, `image_url`,
or `image_base64` is set.

If a request includes SGLang launch-time fields such as `request.options.num_inference_steps`,
`request.options.seed`, `request.options.attention_backend`, `request.options.vsa_sparsity`, non-default
`request.options.num_gpus`, or `request.options.provider_options`, the local provider rejects it. Configure those
values on `scripts/serve_sglang_diffusion.sh`.

Local generation should go through the native SGLang Diffusion server. If a job log does not start with
`SGLang server:` and `POST /v1/videos`, the backend process is stale.

Start SGLang with `scripts/serve_sglang_diffusion.sh`, not by calling the installed SGLang entrypoint directly. The
project launcher applies the pinned SGLang 0.5.x `/v1/videos` output-filename compatibility patch before starting the
native diffusion server.

If the server is stale after code changes, restart the tmux server process and re-check:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/video/providers | python -m json.tool
```

## Validation

Run fast tests after backend or config changes:

```bash
source .venv/bin/activate
python -m pytest tests/test_video_backend.py tests/test_worldodyssey_adapter.py tests/test_wan_benchmark.py -q
python -m compileall worldodyssey_inference scripts tests
git diff --check
```

Full I2V generation is a slow validation and should start with low-frame foreground testing before production tmux runs.
