from __future__ import annotations

import os
from pathlib import Path

import pytest

from worldodyssey_inference.fastvideo_compat import configure_fastvideo_torch_compat
from worldodyssey_inference.tiny_wan import (
    build_tiny_wan_pipeline,
    load_source_tokenizer,
    resolve_recipe,
    save_pipeline,
    verify_saved_pipeline,
)


pytestmark = pytest.mark.slow

RECIPE_NAME = "fastwan2.1-t2v-dmd"
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "artifacts/tiny-fastwan2.1-t2v-dmd-debug"
OUTPUT_PATH = REPO_ROOT / "artifacts/backend-videos/fastvideo_tiny_fastwan.mp4"
PROMPT = "A camera glides over a quiet city street at night."


def configure_fastvideo_runtime() -> None:
    gcc = Path("/usr/bin/gcc")
    gxx = Path("/usr/bin/g++")
    if not gcc.exists() or not gxx.exists():
        pytest.skip("FastVideo runtime requires system gcc and g++ at /usr/bin.")

    os.environ["CC"] = str(gcc)
    os.environ["CXX"] = str(gxx)
    os.environ["PATH"] = f"{gcc.parent}:{os.environ['PATH']}"
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"


def ensure_tiny_fastwan_artifact() -> None:
    recipe = resolve_recipe(RECIPE_NAME)
    if not (MODEL_PATH / "model_index.json").is_file():
        tokenizer = load_source_tokenizer(recipe)
        pipeline = build_tiny_wan_pipeline(recipe=recipe, tokenizer=tokenizer)
        save_pipeline(
            pipeline=pipeline,
            output_dir=MODEL_PATH,
            overwrite=True,
            recipe=recipe,
            repo_id="LOCAL_TINY_FASTWAN",
        )

    verification_message = verify_saved_pipeline(output_dir=MODEL_PATH, recipe=recipe)
    assert verification_message == "FastVideo metadata check passed"


def changed_video_paths(before: dict[Path, int]) -> list[Path]:
    paths = sorted(OUTPUT_PATH.parent.glob(f"{OUTPUT_PATH.stem}*.mp4"))
    return [
        path
        for path in paths
        if path not in before or path.stat().st_mtime_ns != before[path]
    ]


def test_fastvideo_tiny_fastwan_generates_persistent_video() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("FastVideo tiny FastWan generation requires CUDA.")

    ensure_tiny_fastwan_artifact()
    configure_fastvideo_runtime()
    configure_fastvideo_torch_compat()
    pytest.importorskip("fastvideo")
    from fastvideo import VideoGenerator

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    before = {
        path: path.stat().st_mtime_ns
        for path in OUTPUT_PATH.parent.glob(f"{OUTPUT_PATH.stem}*.mp4")
    }

    generator = VideoGenerator.from_pretrained(
        str(MODEL_PATH),
        num_gpus=1,
    )
    try:
        generator.generate_video(
            PROMPT,
            output_path=str(OUTPUT_PATH),
            save_video=True,
            height=64,
            width=64,
            num_frames=5,
            guidance_scale=1.0,
        )
    finally:
        generator.shutdown()

    videos = changed_video_paths(before)
    assert videos
    assert videos[-1].stat().st_size > 0
