from __future__ import annotations

import os
from pathlib import Path

import pytest

from worldodyssey_inference.fastvideo_compat import configure_fastvideo_torch_compat


pytestmark = pytest.mark.slow

MODEL_ID = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "artifacts/backend-videos/fastvideo_example.mp4"
PROMPT = "A camera glides over a quiet city street at night."


def configure_fastvideo_runtime() -> None:
    gcc = Path("/usr/bin/gcc")
    gxx = Path("/usr/bin/g++")
    if not gcc.exists() or not gxx.exists():
        pytest.skip("FastVideo Triton kernels require system gcc and g++ at /usr/bin.")

    os.environ["CC"] = str(gcc)
    os.environ["CXX"] = str(gxx)
    os.environ["PATH"] = f"{gcc.parent}:{os.environ['PATH']}"
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"


def changed_video_paths(before: dict[Path, int]) -> list[Path]:
    paths = sorted(OUTPUT_PATH.parent.glob(f"{OUTPUT_PATH.stem}*.mp4"))
    return [
        path
        for path in paths
        if path not in before or path.stat().st_mtime_ns != before[path]
    ]


def test_fastvideo_example_generates_persistent_video() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("FastVideo generation requires CUDA.")

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
        MODEL_ID,
        num_gpus=1,
    )
    try:
        generator.generate_video(
            PROMPT,
            output_path=str(OUTPUT_PATH),
            save_video=True,
        )
    finally:
        generator.shutdown()

    videos = changed_video_paths(before)
    assert videos
    assert videos[-1].stat().st_size > 0
