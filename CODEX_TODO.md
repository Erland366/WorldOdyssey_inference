# Codex handoff

Continue work in `/workspace/WorldOdyssey_inference`.

## Goal

Provide one default installation that supports Cosmos 3, FastWan, and the
other SGLang Diffusion models through the same provider-neutral video API.

## Completed

- The unified GPU environment is `.venv_sglang`; the API environment is
  `.venv`.
- The separate `.venv_cosmos` and `.venv_fastvideo` environments are
  deprecated and are no longer created or used by the installation scripts.
- Cosmos 3 now runs through native SGLang Diffusion and the unified
  `/v1/video/generations` backend.
- Cosmos 3 passed an end-to-end generation test. The result is
  `artifacts/backend-videos/cosmos3-sglang-unified-smoke.mp4`.
- The unified installer passed with these pins:
  - Torch `2.11.0+cu128`
  - SGLang wheel metadata `0.5.10.post1`
  - SGLang source `9450696aa5dac5fd49a411786e164402fcdee9f5`
  - `sglang-kernel` `0.4.3+cu129`
  - Transformers `5.8.1`
  - `cosmos-guardrail` `0.3.1`
  - FastVideo/VSA source `055e52e5ea44b30cf982a3f248baa36ef386a39c`
- FastWan's VSA extension was rebuilt successfully against Torch 2.11 and
  CUDA 12.8.
- FastWan reached native SGLang startup with
  `attention_backend=video_sparse_attn` and
  `attention_backend_config={"VSA_sparsity": 0.5}`.
- The public backend and SGLang server were stopped cleanly before shutdown.

## Primary TODO

Complete the FastWan end-to-end smoke test.

1. Start the native FastWan server:

   ```bash
   bash scripts/serve_sglang_diffusion.sh \
     FastVideo/FastWan2.1-T2V-1.3B-Diffusers \
     --attention-backend video_sparse_attn \
     --VSA-sparsity 0.5
   ```

2. Wait for the warmup and port 30000 to become ready. SGLang module startup
   can take several minutes on this host.

3. Start the public backend in another terminal:

   ```bash
   .venv/bin/python scripts/serve_video_backend.py
   ```

4. Submit a tiny request to `/v1/video/generations` using:
   - provider: `sglang`
   - model: `FastVideo/FastWan2.1-T2V-1.3B-Diffusers`
   - width: `320`
   - height: `192`
   - frames: `5`
   - inference steps: `1`
   - fps: `16`
   - timeout: `600`

5. Poll the job, download its MP4, and validate the codec, dimensions, FPS,
   and frame count with `ffprobe`.

6. If VSA fails at inference time, diagnose the exact runtime exception. Do
   not replace it with the PyPI VSA wheel: that wheel has the wrong Torch ABI
   for this environment. Its locally built import and SGLang argument parsing
   already pass.

7. Stop both persistent services after validation.

## Final verification

Run:

```bash
.venv/bin/python -m pytest -m 'not slow' -q
git diff --check
git status --short
```

Also run `bash -n` on every modified shell script. Review all changes without
overwriting unrelated user edits. The existing `cloudpickle` change in
`pyproject.toml` may predate this work.

Search for stale instructions that still prescribe `.venv_cosmos`, three
runtime environments, or direct Cosmos Diffusers execution. Historical and
explicit deprecation notes are acceptable. After FastWan passes, update the
README validation status and ensure `references/sglang-diffusion.md` explains
that VSA is pinned by source revision and built locally.

## Architecture to preserve

- `scripts/serve_sglang_diffusion.sh` starts the native model server.
- `scripts/serve_video_backend.py` provides job management, polling, storage,
  and the unified public API.
- `scripts/run_cosmos3.sh` is a client helper, not a third service.
- A single SGLang process loads one model at a time. Cosmos 3 and FastWan
  nevertheless use the same installation, environment, and public API.

## Important files

- `runtime-versions.env`
- `install.sh`
- `scripts/install_sglang_diffusion.sh`
- `scripts/install_cosmos3.sh`
- `scripts/serve_sglang_diffusion.sh`
- `scripts/serve_video_backend.py`
- `scripts/run_cosmos3.py`
- `scripts/run_cosmos3.sh`
- `worldodyssey_inference/video_backend/providers.py`
- `README.md`
- `docs/cosmos3.md`
- `docs/video-backend-runbook.md`
- `references/sglang-diffusion.md`
