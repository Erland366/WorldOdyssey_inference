from __future__ import annotations

import os
import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import pytest

from worldodyssey_inference.tiny_wan import (
    build_tiny_wan_pipeline,
    load_source_tokenizer,
    resolve_recipe,
    save_pipeline,
    verify_saved_pipeline,
)


pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "artifacts/backend-videos"
PROMPT = "A camera glides over a quiet city street at night."


@dataclass(frozen=True)
class SGLangCase:
    recipe_name: str
    model_path: Path
    model_id: str
    output_file_name: str
    extra_args: tuple[str, ...] = ()


SGLANG_CASES = (
    SGLangCase(
        recipe_name="wan2.1-t2v-1.3b",
        model_path=REPO_ROOT / "artifacts/tiny-wan2.1-t2v-debug",
        model_id="Wan2.1-T2V-1.3B-Diffusers",
        output_file_name="sglang_tiny_wan_native.mp4",
    ),
    SGLangCase(
        recipe_name="fastwan2.1-t2v-dmd",
        model_path=REPO_ROOT / "artifacts/tiny-fastwan2.1-t2v-dmd-debug",
        model_id="FastWan2.1-T2V-1.3B-Diffusers",
        output_file_name="sglang_tiny_fastwan_native.mp4",
        extra_args=("--dmd-denoising-steps", "1000"),
    ),
)


def require_sglang_cli() -> str:
    sglang = shutil.which("sglang")
    if sglang is None:
        pytest.skip(
            "SGLang Diffusion is not installed in this virtualenv. "
            'Install with `uv pip install "sglang[diffusion]" --prerelease=allow`.'
        )
    return sglang


def ensure_tiny_artifact(case: SGLangCase) -> None:
    recipe = resolve_recipe(case.recipe_name)
    needs_rebuild = not (case.model_path / "model_index.json").is_file()
    if recipe.diffusers_smoke_test and not (case.model_path / "tokenizer/spiece.model").is_file():
        needs_rebuild = True

    if needs_rebuild:
        tokenizer = load_source_tokenizer(recipe)
        pipeline = build_tiny_wan_pipeline(recipe=recipe, tokenizer=tokenizer)
        save_pipeline(
            pipeline=pipeline,
            output_dir=case.model_path,
            overwrite=case.model_path.exists(),
            recipe=recipe,
            repo_id=f"LOCAL_{case.recipe_name.upper()}",
        )

    verification_message = verify_saved_pipeline(output_dir=case.model_path, recipe=recipe)
    if recipe.diffusers_smoke_test:
        assert verification_message.startswith("Diffusers smoke test output shape:")
    else:
        assert verification_message == "FastVideo metadata check passed"


def sglang_env() -> dict[str, str]:
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    site_packages = Path(sysconfig.get_paths()["purelib"])
    nvidia_root = site_packages / "nvidia"
    cuda_library_dirs = [
        nvidia_root / "cuda_runtime/lib",
        nvidia_root / "nccl/lib",
        nvidia_root / "cublas/lib",
        nvidia_root / "cuda_nvrtc/lib",
        nvidia_root / "nvjitlink/lib",
        nvidia_root / "cusparse/lib",
        nvidia_root / "cusolver/lib",
        nvidia_root / "cudnn/lib",
        nvidia_root / "cufft/lib",
        nvidia_root / "curand/lib",
        nvidia_root / "cufile/lib",
        nvidia_root / "nvtx/lib",
        nvidia_root / "cu13/lib",
    ]
    existing_library_path = env.get("LD_LIBRARY_PATH")
    library_path_parts = [str(path) for path in cuda_library_dirs if path.is_dir()]
    if existing_library_path:
        library_path_parts.append(existing_library_path)
    if library_path_parts:
        env["LD_LIBRARY_PATH"] = ":".join(library_path_parts)

    return env


def run_sglang_generate(case: SGLangCase, sglang: str) -> subprocess.CompletedProcess[str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / case.output_file_name
    command = [
        sglang,
        "generate",
        "--model-path",
        str(case.model_path),
        "--model-id",
        case.model_id,
        "--backend",
        "sglang",
        "--prompt",
        PROMPT,
        "--height",
        "64",
        "--width",
        "64",
        "--num-frames",
        "5",
        "--num-inference-steps",
        "1",
        "--guidance-scale",
        "1.0",
        "--dit-precision",
        "fp32",
        "--vae-precision",
        "fp32",
        "--output-path",
        str(OUTPUT_DIR),
        "--output-file-name",
        output_path.name,
        "--save-output",
        *case.extra_args,
    ]
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=sglang_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize("case", SGLANG_CASES, ids=lambda case: case.recipe_name)
def test_sglang_native_wan_generates_persistent_video(case: SGLangCase) -> None:
    sglang = require_sglang_cli()
    ensure_tiny_artifact(case)
    result = run_sglang_generate(case, sglang)
    output_path = OUTPUT_DIR / case.output_file_name

    assert result.returncode == 0, result.stdout
    assert output_path.exists()
    assert output_path.stat().st_size > 0
