---
name: fastvideo-tiny-fixtures
description: Create, debug, and validate tiny FastVideo-compatible Wan/FastWan/VSA video-generation fixtures. Use when working on FastVideo `VideoGenerator.from_pretrained(...)` loading, tiny Wan debug models, FastWan DMD fixtures, VSA fixtures, backend-specific FastVideo metadata, or slow backend tests that generate persistent videos.
---

# FastVideo Tiny Fixtures

## Purpose

Use this skill to create tiny video-generation artifacts that exercise the real FastVideo loading and runtime paths without running full-size Wan checkpoints.

The goal is not visual quality. The goal is a small artifact that preserves the relevant component layout, metadata, backend constraints, and runtime-only parameters needed to catch integration bugs quickly.

## Workflow

1. Inspect the upstream model before changing the tiny recipe:
   - `model_index.json`
   - transformer config
   - scheduler config
   - component names
   - FastVideo model registry handling for the upstream repo id

2. Determine the actual runtime class and backend:
   - Check whether FastVideo exact-matches the upstream model id.
   - Check whether the local or Hugging Face tiny repo will fall back to a generic config.
   - Check `FASTVIDEO_ATTENTION_BACKEND`.
   - Check whether `pipeline_config` must be passed explicitly.

3. Do not assume the public Diffusers architecture is the full FastVideo runtime architecture.
   FastVideo can select backend-specific blocks or add runtime-only modules while the saved model still looks like a standard Diffusers component tree.

4. Keep the fixture tiny, but respect backend invariants:
   - Keep frame count, height, width, layer count, and FFN size small.
   - Do not shrink dimensions below kernel/backend constraints.
   - Prefer a shape that runs through the target backend over the smallest possible tensor shape.

5. Save and validate the artifact:
   - Remove stale output directories when using overwrite.
   - Save with `save_pretrained`.
   - Patch backend-specific metadata or weights after saving.
   - Verify required component weight files are discoverable by glob-based loaders.
   - Validate with fast tests first, then slow FastVideo backend tests.

## FastWan DMD Pattern

For FastWan DMD tiny fixtures:

- Use `WanDMDPipeline` metadata.
- Use FastVideo loading as the compatibility check.
- Do not require a Diffusers `WanPipeline` smoke test for the saved artifact.
- Use `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA` unless the target test explicitly needs another backend.

The slow acceptance surface should call `VideoGenerator.from_pretrained(...)` and generate a short persistent video under `artifacts/backend-videos/`.

## VSA Pattern

For VSA tiny fixtures:

- Use `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`.
- Use a backend-supported attention head dimension, usually `64` or `128`.
- Pass `pipeline_config={"flow_shift": 5.0}` when loading a local or personal Hugging Face tiny repo that does not hit FastVideo's exact upstream registry match.
- Pass the intended `VSA_sparsity` value in the FastVideo call.
- Ensure each transformer block has serialized `to_gate_compress.weight` and `to_gate_compress.bias` tensors.

Zero-initialized `to_gate_compress` tensors are acceptable for random tiny debug fixtures. Their purpose is compatibility and warning-free loading, not meaningful generation quality.

## Validation

Run fast tests before slow backend tests:

```bash
source .venv/bin/activate && python -m pytest -m "not slow" -q
```

Run targeted slow tests only when the backend acceptance surface matters:

```bash
source .venv/bin/activate && python -m pytest tests/backends -m slow -q -s
```

Generated videos should remain on disk under `artifacts/backend-videos/` for manual inspection.

## Known Failure Modes

- `Found unloaded parameters ... to_gate_compress`
  - Cause: VSA runtime expects FastVideo-only gate-compression weights.
  - Fix: Patch tiny transformer safetensors with explicit zero tensors.

- Invalid VSA attention head size
  - Cause: `VIDEO_SPARSE_ATTN` does not support arbitrary tiny head dimensions.
  - Fix: Use a supported head dimension such as `64` or `128`.

- Wrong FastVideo config fallback
  - Cause: Local or personal Hugging Face tiny repo ids may not match FastVideo's upstream registry entries.
  - Fix: Pass required `pipeline_config` explicitly.

- No safetensors discovered after saving
  - Cause: FastVideo discovers component weights by globbing, which can lag behind direct file visibility on shared filesystems.
  - Fix: Clean stale output directories before saving and refresh component directories after saving.

- Triton or FastVideo compiler failures
  - Cause: Conda compiler wrappers can be selected instead of system compilers.
  - Fix: Prefer `CC=/usr/bin/gcc CXX=/usr/bin/g++ PATH=/usr/bin:$PATH` for FastVideo slow tests.
