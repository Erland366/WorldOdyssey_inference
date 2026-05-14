from __future__ import annotations

from pathlib import Path

import pytest

from worldodyssey_inference.tiny_wan import (
    build_tiny_wan_pipeline,
    load_source_tokenizer,
    resolve_recipe,
    save_pipeline,
)


pytestmark = pytest.mark.slow

RECIPE_NAME = "wan2.1-t2v-1.3b"
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "artifacts/tiny-wan2.1-t2v-debug"
OUTPUT_PATH = REPO_ROOT / "artifacts/backend-videos/diffusers_example.mp4"
PROMPT = "A camera glides over a quiet city street at night."
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


def ensure_tiny_wan_artifact() -> None:
    recipe = resolve_recipe(RECIPE_NAME)
    needs_rebuild = not (MODEL_PATH / "model_index.json").is_file()
    needs_rebuild = needs_rebuild or not (MODEL_PATH / "tokenizer/spiece.model").is_file()

    if not needs_rebuild:
        return

    tokenizer = load_source_tokenizer(recipe)
    pipeline = build_tiny_wan_pipeline(recipe=recipe, tokenizer=tokenizer)
    save_pipeline(
        pipeline=pipeline,
        output_dir=MODEL_PATH,
        overwrite=MODEL_PATH.exists(),
        recipe=recipe,
        repo_id="LOCAL_TINY_WAN",
    )


def test_diffusers_example_generates_persistent_video() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")

    if not torch.cuda.is_available():
        pytest.skip("Diffusers Wan video generation requires CUDA.")

    pytest.importorskip("diffusers")
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ensure_tiny_wan_artifact()
    pipe = WanPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype={
            "default": torch.bfloat16,
            "vae": torch.float32,
        },
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    frames = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=1,
        guidance_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(1024),
        max_sequence_length=8,
    ).frames[0]

    export_to_video(frames, str(OUTPUT_PATH), fps=8)

    assert OUTPUT_PATH.exists()
    assert OUTPUT_PATH.stat().st_size > 0
