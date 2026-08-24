# MiniMax-H3 through SGLang

MiniMax-H3 is isolated from the existing FastWan/Cosmos environment:

- `.venv_sglang` remains on the validated CUDA 12 SGLang source revision.
- `.venv_sglang_h3` uses pinned SGLang `0.5.18` and its CUDA 13 dependencies.
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

The model itself is downloaded on first launch. Review and accept the MiniMax-H3 model license before use.

## FL2VA: text or endpoint images

Start the primary server:

```bash
bash scripts/run_minimax_h3_fl2va.sh
```

The default single-GPU launch uses SGLang's lossless layerwise-offload recipe. Override topology variables when using
multiple GPUs, for example:

```bash
WORLDODYSSEY_MINIMAX_H3_NUM_GPUS=4 \
WORLDODYSSEY_MINIMAX_H3_TP_SIZE=2 \
WORLDODYSSEY_MINIMAX_H3_ULYSSES_DEGREE=2 \
WORLDODYSSEY_MINIMAX_H3_PERFORMANCE_MODE=speed \
bash scripts/run_minimax_h3_fl2va.sh
```

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
