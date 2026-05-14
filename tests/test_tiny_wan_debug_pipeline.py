from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import load_file as load_safetensors_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from worldodyssey_inference.tiny_wan import (
    build_tiny_wan_pipeline,
    resolve_recipe,
    run_smoke_test,
    save_pipeline,
    verify_fastvideo_metadata,
    verify_saved_pipeline,
)


def make_tiny_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "</s>": 1,
                "<unk>": 2,
                "debug": 3,
                "prompt": 4,
            },
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )


@pytest.mark.slow
@pytest.mark.parametrize("recipe_name", ["wan2.1-t2v-1.3b", "wan2.2-t2v-a14b"])
def test_tiny_wan_pipeline_round_trips_and_exports(tmp_path: Path, recipe_name: str) -> None:
    recipe = resolve_recipe(recipe_name)
    pipeline = build_tiny_wan_pipeline(recipe=recipe, tokenizer=make_tiny_tokenizer())
    output_dir = tmp_path / "tiny-wan"

    save_pipeline(pipeline=pipeline, output_dir=output_dir, overwrite=False, recipe=recipe)

    assert run_smoke_test(output_dir) == (5, 64, 64, 3)


def test_save_pipeline_fails_when_output_exists(tmp_path: Path) -> None:
    recipe = resolve_recipe("wan2.1-t2v-1.3b")
    pipeline = build_tiny_wan_pipeline(recipe=recipe, tokenizer=make_tiny_tokenizer())
    output_dir = tmp_path / "tiny-wan"

    save_pipeline(pipeline=pipeline, output_dir=output_dir, overwrite=False, recipe=recipe)

    with pytest.raises(FileExistsError, match="--overwrite"):
        save_pipeline(pipeline=pipeline, output_dir=output_dir, overwrite=False, recipe=recipe)

    save_pipeline(pipeline=pipeline, output_dir=output_dir, overwrite=True, recipe=recipe)


def test_save_pipeline_overwrite_rebuilds_clean_directory(tmp_path: Path) -> None:
    recipe = resolve_recipe("fastwan2.1")
    pipeline = build_tiny_wan_pipeline(recipe=recipe, tokenizer=make_tiny_tokenizer())
    output_dir = tmp_path / "tiny-fastwan-dmd"
    stale_marker = output_dir / "stale.txt"

    (output_dir / "transformer").mkdir(parents=True)
    (output_dir / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    stale_marker.write_text("stale", encoding="utf-8")

    save_pipeline(pipeline=pipeline, output_dir=output_dir, overwrite=True, recipe=recipe)

    assert not stale_marker.exists()
    assert any((output_dir / "transformer").glob("*.safetensors"))
    verify_fastvideo_metadata(output_dir=output_dir, recipe=recipe)


def test_wan22_recipe_builds_two_transformers() -> None:
    recipe = resolve_recipe("wan2.2")
    pipeline = build_tiny_wan_pipeline(recipe=recipe, tokenizer=make_tiny_tokenizer())

    assert pipeline.transformer is not None
    assert pipeline.transformer_2 is not None
    assert pipeline.config.boundary_ratio == 0.875


def test_fastwan_dmd_recipe_writes_fastvideo_model_index(tmp_path: Path) -> None:
    recipe = resolve_recipe("fastwan2.1")
    pipeline = build_tiny_wan_pipeline(recipe=recipe, tokenizer=make_tiny_tokenizer())
    output_dir = tmp_path / "tiny-fastwan-dmd"

    save_pipeline(pipeline=pipeline, output_dir=output_dir, overwrite=False, recipe=recipe)

    model_index = json.loads((output_dir / "model_index.json").read_text(encoding="utf-8"))
    assert model_index["_class_name"] == "WanDMDPipeline"
    assert model_index["transformer"] == ["diffusers", "WanTransformer3DModel"]
    assert model_index["vae"] == ["diffusers", "AutoencoderKLWan"]
    assert "transformer_2" not in model_index
    assert "boundary_ratio" not in model_index
    assert "expand_timesteps" not in model_index

    verify_fastvideo_metadata(output_dir=output_dir, recipe=recipe)
    assert verify_saved_pipeline(output_dir=output_dir, recipe=recipe) == "FastVideo metadata check passed"


def test_vsa_recipe_writes_fastvideo_sparse_attention_fixture(tmp_path: Path) -> None:
    recipe = resolve_recipe("vsa")
    pipeline = build_tiny_wan_pipeline(recipe=recipe, tokenizer=make_tiny_tokenizer())
    output_dir = tmp_path / "tiny-vsa"

    save_pipeline(pipeline=pipeline, output_dir=output_dir, overwrite=False, recipe=recipe)

    model_index = json.loads((output_dir / "model_index.json").read_text(encoding="utf-8"))
    transformer_config = json.loads((output_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    transformer_state_dict = load_safetensors_file(str(output_dir / "transformer" / "diffusion_pytorch_model.safetensors"))

    assert model_index["_class_name"] == "WanPipeline"
    assert model_index["transformer"] == ["diffusers", "WanTransformer3DModel"]
    assert transformer_config["attention_head_dim"] == 64
    assert transformer_config["num_attention_heads"] == 1
    assert tuple(transformer_state_dict["blocks.0.to_gate_compress.weight"].shape) == (64, 64)
    assert tuple(transformer_state_dict["blocks.0.to_gate_compress.bias"].shape) == (64,)
    assert recipe.fastvideo_backend == "VIDEO_SPARSE_ATTN"

    verify_fastvideo_metadata(output_dir=output_dir, recipe=recipe)
    assert verify_saved_pipeline(output_dir=output_dir, recipe=recipe) == "FastVideo metadata check passed"
