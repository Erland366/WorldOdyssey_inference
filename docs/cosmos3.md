# Cosmos 3 through SGLang

Cosmos 3 and FastWan use one installation, one SGLang runtime, and one WorldOdyssey API:

```bash
./install.sh
```

The GPU runtime is `.venv_sglang`. `runtime-versions.env` pins its CUDA-12.8 Torch stack, CUDA-12.9 SGLang kernel,
Cosmos-capable SGLang source commit, and `cosmos-guardrail` version. The direct `.venv_cosmos` Diffusers path is
deprecated; `scripts/install_cosmos3.sh` is now only a compatibility wrapper for the unified installer.

## Serve and submit

One native SGLang process loads one checkpoint. Start Cosmos 3 Nano:

```bash
bash scripts/run_cosmos3.sh
```

Start the provider-neutral backend in another shell:

```bash
bash scripts/run_backend.sh
```

Submit Cosmos through `POST /v1/video/generations` with the helper:

```bash
bash scripts/generate_cosmos3.sh
```

The helper submits, polls, and downloads the result. A small smoke request is:

```bash
bash scripts/generate_cosmos3.sh \
  --no-guardrails \
  --num-frames 5 \
  --height 192 \
  --width 320 \
  --steps 1 \
  --output artifacts/backend-videos/cosmos3-smoke.mp4
```

For image-to-video, combine the prompt with exactly one local image or image URL:

```bash
bash scripts/generate_cosmos3.sh \
  --image-path images/planet.png \
  --prompt "The camera flies toward the planet while its rings shimmer." \
  --output artifacts/backend-videos/planet.mp4
```

The copy-ready wrapper in `examples/generate_cosmos3_i2v.sh` accepts the image as its first argument. Set
`COSMOS3_I2V_PROMPT` and `COSMOS3_I2V_OUTPUT` to customize it without editing the file.

The request uses the same schema as FastWan. Cosmos-specific values are forwarded under
`options.provider_options`: `flow_shift` as a multipart request field and guardrail/template flags as
`extra_params`. `--generate-audio` maps to native SGLang's `generate_sound` field.

Guardrails are enabled by default and require access to NVIDIA's gated guardrail weights. For a deliberate local run
without them, `run_cosmos3.sh` disables server-side guardrails by default; pass `--no-guardrails` to the generation
helper as well. Set `SGLANG_DISABLE_COSMOS3_GUARDRAILS=0` before the server launcher when gated weights are available.

To switch back to FastWan, stop only the native SGLang process and restart it with the FastWan model. The backend URL
and `/v1/video/generations` contract do not change.
