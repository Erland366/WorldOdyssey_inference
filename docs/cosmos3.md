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
SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
bash scripts/serve_sglang_diffusion.sh nvidia/Cosmos3-Nano
```

Start the provider-neutral backend in another shell:

```bash
export WORLDODYSSEY_SGLANG_BASE_URL=http://127.0.0.1:30000
source .venv/bin/activate
python scripts/serve_video_backend.py --host 127.0.0.1 --port 8000
```

Submit Cosmos through `POST /v1/video/generations` with the helper:

```bash
bash scripts/run_cosmos3.sh
```

The helper submits, polls, and downloads the result. A small smoke request is:

```bash
bash scripts/run_cosmos3.sh \
  --no-guardrails \
  --num-frames 5 \
  --height 192 \
  --width 320 \
  --steps 1 \
  --output artifacts/backend-videos/cosmos3-smoke.mp4
```

The request uses the same schema as FastWan. Cosmos-specific values are forwarded under
`options.provider_options`: `flow_shift` as a multipart request field and guardrail/template flags as
`extra_params`. `--generate-audio` maps to native SGLang's `generate_sound` field.

Guardrails are enabled by default and require access to NVIDIA's gated guardrail weights. For a deliberate local run
without them, set `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` on the native server and pass `--no-guardrails` to the helper.

To switch back to FastWan, stop only the native SGLang process and restart it with the FastWan model. The backend URL
and `/v1/video/generations` contract do not change.
