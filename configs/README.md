# Inference Submission Configs

These YAML files drive the local video backend submitters:

- `worldodyssey-move-bookmark-t2v.yaml`: WorldOdyssey task-to-video request.
- `worldodyssey-move-bookmark-i2v.yaml`: WorldOdyssey image-to-video request using the task's main frame.
- `worldodyssey-move-bookmark-t2v-hunyuan-fp8-visual.yaml`: single WorldOdyssey `move_bookmark` T2V run using
  Hunyuan FP8 at supported 540p dimensions for visual inspection.
- `worldodyssey-inputs-batch-t2v.yaml`: WorldOdyssey parent `inputs/` directory submitted as one T2V batch.
- `worldodyssey-inputs-batch-t2v-tinywan.yaml`: WorldOdyssey parent `inputs/` batch using the tiny Wan debug model.
- `worldodyssey-inputs-batch-t2v-hunyuan-fp8.yaml`: WorldOdyssey parent `inputs/` batch using the Hunyuan FP8
  multipart SGLang server path.
- `worldodyssey-inputs-batch-t2v-wan-InP.yaml`: WorldOdyssey parent `inputs/` batch in I2V mode with
  `weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers` and low-memory five-frame smoke dimensions.
- `tiny-wan-batch.yaml`: provider-neutral batch request using `Erland/tiny-wan2.1-t2v-debug`.

Run one directly:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-move-bookmark-i2v.yaml
```

Before submitting, the native SGLang server and the provider-neutral backend must already be running. The backend needs
`WORLDODYSSEY_SGLANG_BASE_URL`; `WORLDODYSSEY_SGLANG_VIDEO_API_FORMAT` defaults to the unified native `multipart`
server path. The backend forwards each config's `request.model` to SGLang and lets SGLang handle model compatibility.
WorldOdyssey configs assume the `submodule/worldodyssey` git submodule has been initialized. `bash
scripts/setup_video_backend.sh` does this during setup; for input-only work, run:

```bash
git submodule update --init --recursive submodule/worldodyssey
```

Override any value from the command line with dotted paths:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-i2v.yaml \
  --set request.options.num_frames=17 \
  --set run.download_path=artifacts/video-backend/test-i2v.mp4
```

Config sections:

- `task`: WorldOdyssey task directory, `task.json`, or a parent directory containing task folders.
- `backend_url`: video backend URL.
- `adapter`: WorldOdyssey adapter controls, such as `i2v`, `include_main_image_base64`, and `prompt_prefix`.
- `request`: canonical `/v1/video/generations` request fields.
- `run`: submitter behavior, such as `dry_run`, `wait`, `download_path`, and `download_dir`.

By default, the WorldOdyssey prompt is only the `task` field from `task.json`. Use `adapter.prompt_prefix` to prepend
extra inference instruction text without replacing the task.

For I2V, the WorldOdyssey adapter attaches the task's main frame as `image_base64` by default. To use a custom local
image path instead, disable that attachment and set `request.image_path`:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-i2v.yaml \
  --set adapter.include_main_image_base64=false \
  --set request.image_path=/absolute/path/to/input.png \
  --set request.options.num_frames=17 \
  --set run.download_path=artifacts/video-backend/custom-image-i2v-smoke.mp4
```

The local SGLang provider accepts exactly one image input: `request.image_path`, `request.image_url`, or
`request.image_base64`.

Specific CLI flags such as `--height`, `--num-frames`, and `--dry-run` override the YAML. `--set` is applied last.

## WorldOdyssey Parent Inputs

`scripts/submit_worldodyssey_task.py` accepts a parent directory such as:

```text
submodule/worldodyssey/inputs
```

The submitter scans direct child folders with `task.json` and submits them as one
`POST /v1/video/generation-batches` request. Child folders without `task.json` are not submitted; their paths are
recorded in batch metadata as `skipped_entries`.

Dry-run the imported parent directory:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  submodule/worldodyssey/inputs \
  --dry-run
```

Or use the checked-in parent-input batch config:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-inputs-batch-t2v.yaml \
  --dry-run
```

For parent-directory batches, use `run.download_dir` or `--download-dir`:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py --config configs/worldodyssey-inputs-batch-t2v.yaml
```

`run.download_path` is only valid for single-task submissions.

## Batch Configs

Batch configs are submitted with `scripts/submit_video_batch.py`. Their root mirrors
`POST /v1/video/generation-batches`, plus `backend_url` and `run` controls:

```bash
source .venv/bin/activate
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml --dry-run
```

Submit and wait:

```bash
source .venv/bin/activate
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml
```

Override individual batch items with numeric list indexes:

```bash
source .venv/bin/activate
python scripts/submit_video_batch.py configs/tiny-wan-batch.yaml \
  --set 'requests.0.prompt=A tiny camera circles a glass cube.' \
  --set requests.1.options.num_frames=5
```

The local SGLang backend talks to a persistent native SGLang Diffusion server through `/v1/videos`. Config files always
carry the provider-neutral request shape: prompt, model, mode, image input, `height`, `width`, `num_frames`, optional
`duration`, optional `fps`, and timeouts.

The unified multipart local provider forwards `num_inference_steps`, `seed`, `guidance_scale`, `negative_prompt`,
scalar `provider_options.request_fields`, and structured `provider_options.extra_body`. It rejects launch-time knobs:
non-default `num_gpus`, `attention_backend`, and `vsa_sparsity`. Configure GPU count, VSA, tensor parallelism, sequence
parallelism, transformer overrides, and memory offload when starting `scripts/serve_sglang_diffusion.sh`.

For Hunyuan FP8, keep `request.model` as `hunyuanvideo-community/HunyuanVideo`. Pass
`lmsys/hunyuanvideo-modelopt-fp8-sglang-transformer` to SGLang at server launch with `--transformer-path`.

Use the checked-in visual config when you want an inspectable single output:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-move-bookmark-t2v-hunyuan-fp8-visual.yaml
```

It writes:

```text
artifacts/video-backend/worldodyssey-move-bookmark-hunyuan-fp8-visual.mp4
```

When debugging a failed run, the job log should start with:

```text
SGLang server: http://127.0.0.1:30000
POST /v1/videos
```

For the WorldOdyssey parent-input adapter with tinywan, use:

```bash
source .venv/bin/activate
python scripts/submit_worldodyssey_task.py \
  --config configs/worldodyssey-inputs-batch-t2v-tinywan.yaml \
  --dry-run
```
