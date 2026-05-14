"""Tiny random-weight Wan pipeline fixtures for fast debugging."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
from huggingface_hub import HfApi
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from transformers import AutoTokenizer, PreTrainedTokenizerBase, UMT5Config, UMT5EncoderModel


TEXT_DIM = 32
LATENT_CHANNELS = 4
SMOKE_HEIGHT = 64
SMOKE_WIDTH = 64
SMOKE_FRAMES = 5
SMOKE_STEPS = 1
SMOKE_MAX_SEQUENCE_LENGTH = 8


@dataclass(frozen=True)
class TinyWanRecipe:
    """Configuration for one reusable tiny Wan artifact."""

    name: str
    source_model_id: str
    tokenizer_subfolder: str
    default_output_dir: Path
    default_repo_name: str
    model_card_title: str
    source_description: str
    include_transformer_2: bool = False
    boundary_ratio: float | None = None
    model_index_class_name: str = "WanPipeline"
    diffusers_smoke_test: bool = True
    fastvideo_backend: str | None = None
    fastvideo_flow_shift: float | None = None
    fastvideo_vsa_sparsity: float | None = None
    transformer_num_attention_heads: int = 2
    transformer_attention_head_dim: int = 16
    transformer_ffn_dim: int = 64
    transformer_num_layers: int = 1


RECIPES: dict[str, TinyWanRecipe] = {
    "wan2.1-t2v-1.3b": TinyWanRecipe(
        name="wan2.1-t2v-1.3b",
        source_model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        tokenizer_subfolder="tokenizer",
        default_output_dir=Path("artifacts/tiny-wan2.1-t2v-debug"),
        default_repo_name="tiny-wan2.1-t2v-debug",
        model_card_title="Tiny Wan2.1 T2V Debug Pipeline",
        source_description="Wan2.1 text-to-video Diffusers pipeline with one denoising transformer.",
    ),
    "wan2.2-t2v-a14b": TinyWanRecipe(
        name="wan2.2-t2v-a14b",
        source_model_id="Wan-AI/Wan2.2-T2V-A14B",
        tokenizer_subfolder="google/umt5-xxl",
        default_output_dir=Path("artifacts/tiny-wan2.2-t2v-a14b-debug"),
        default_repo_name="tiny-wan2.2-t2v-a14b-debug",
        model_card_title="Tiny Wan2.2 T2V A14B Debug Pipeline",
        source_description=(
            "Wan2.2 T2V-A14B high/low-noise expert layout represented as a Diffusers "
            "WanPipeline with `transformer` and `transformer_2`."
        ),
        include_transformer_2=True,
        boundary_ratio=0.875,
    ),
    "fastwan2.1-t2v-dmd": TinyWanRecipe(
        name="fastwan2.1-t2v-dmd",
        source_model_id="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        tokenizer_subfolder="tokenizer",
        default_output_dir=Path("artifacts/tiny-fastwan2.1-t2v-dmd-debug"),
        default_repo_name="tiny-fastwan2.1-t2v-dmd-debug",
        model_card_title="Tiny FastWan2.1 T2V DMD Debug Pipeline",
        source_description=(
            "FastVideo FastWan2.1 DMD-style text-to-video layout represented as a Diffusers-format "
            "artifact with `_class_name` patched to `WanDMDPipeline` for FastVideo `VideoGenerator` "
            "load-path debugging."
        ),
        model_index_class_name="WanDMDPipeline",
        diffusers_smoke_test=False,
        fastvideo_backend="TORCH_SDPA",
    ),
    "wan2.1-vsa-t2v-14b-720p": TinyWanRecipe(
        name="wan2.1-vsa-t2v-14b-720p",
        source_model_id="FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers",
        tokenizer_subfolder="tokenizer",
        default_output_dir=Path("artifacts/tiny-wan2.1-vsa-t2v-14b-720p-debug"),
        default_repo_name="tiny-wan2.1-vsa-t2v-14b-720p-debug",
        model_card_title="Tiny Wan2.1 VSA T2V 14B 720P Debug Pipeline",
        source_description=(
            "FastVideo Wan2.1 VSA 14B 720P text-to-video layout represented as a tiny "
            "Diffusers `WanPipeline` artifact for FastVideo `VideoGenerator` and "
            "`VIDEO_SPARSE_ATTN` load-path debugging. The saved transformer weights include "
            "FastVideo VSA `to_gate_compress` tensors, which are not emitted by Diffusers."
        ),
        diffusers_smoke_test=False,
        fastvideo_backend="VIDEO_SPARSE_ATTN",
        fastvideo_flow_shift=5.0,
        fastvideo_vsa_sparsity=0.5,
        transformer_num_attention_heads=1,
        transformer_attention_head_dim=64,
        transformer_ffn_dim=128,
    ),
}

RECIPE_ALIASES = {
    "wan2.1": "wan2.1-t2v-1.3b",
    "wan2.1-t2v": "wan2.1-t2v-1.3b",
    "wan2.2": "wan2.2-t2v-a14b",
    "wan2.2-a14b": "wan2.2-t2v-a14b",
    "wan2.2-t2v": "wan2.2-t2v-a14b",
    "fastwan2.1": "fastwan2.1-t2v-dmd",
    "fastwan2.1-dmd": "fastwan2.1-t2v-dmd",
    "fastwan": "fastwan2.1-t2v-dmd",
    "vsa": "wan2.1-vsa-t2v-14b-720p",
    "wan2.1-vsa": "wan2.1-vsa-t2v-14b-720p",
    "fastvideo-vsa": "wan2.1-vsa-t2v-14b-720p",
}


def resolve_recipe(recipe_name: str) -> TinyWanRecipe:
    """Resolve a recipe name or alias, failing loudly for unknown values."""
    canonical_name = RECIPE_ALIASES.get(recipe_name, recipe_name)
    if canonical_name not in RECIPES:
        valid_names = ", ".join(sorted(RECIPES | RECIPE_ALIASES))
        raise ValueError(f"Unknown tiny Wan recipe {recipe_name!r}. Valid recipes: {valid_names}.")
    return RECIPES[canonical_name]


def load_source_tokenizer(recipe: TinyWanRecipe) -> PreTrainedTokenizerBase:
    """Load the recipe tokenizer so normal prompt strings exercise the expected path."""
    return AutoTokenizer.from_pretrained(recipe.source_model_id, subfolder=recipe.tokenizer_subfolder)


def build_tiny_wan_pipeline(recipe: TinyWanRecipe, tokenizer: PreTrainedTokenizerBase, seed: int = 0) -> WanPipeline:
    """Build a tiny random-weight WanPipeline with real Diffusers component classes."""
    torch.manual_seed(seed)
    transformer = build_tiny_transformer(recipe)
    transformer_2 = None
    if recipe.include_transformer_2:
        torch.manual_seed(seed + 1)
        transformer_2 = build_tiny_transformer(recipe)

    torch.manual_seed(seed + 2)
    vae = build_tiny_vae()
    text_encoder = build_tiny_text_encoder(tokenizer)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=10, shift=3.0)

    pipeline = WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        transformer=transformer,
        transformer_2=transformer_2,
        boundary_ratio=recipe.boundary_ratio,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to("cpu")
    return pipeline


def build_tiny_transformer(recipe: TinyWanRecipe) -> WanTransformer3DModel:
    """Build one tiny Wan denoising transformer."""
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=recipe.transformer_num_attention_heads,
        attention_head_dim=recipe.transformer_attention_head_dim,
        in_channels=LATENT_CHANNELS,
        out_channels=LATENT_CHANNELS,
        text_dim=TEXT_DIM,
        freq_dim=32,
        ffn_dim=recipe.transformer_ffn_dim,
        num_layers=recipe.transformer_num_layers,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=64,
    )


def build_tiny_vae() -> AutoencoderKLWan:
    """Build the small Wan VAE used by all tiny T2V recipes."""
    return AutoencoderKLWan(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=LATENT_CHANNELS,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        latents_mean=[0.0] * LATENT_CHANNELS,
        latents_std=[1.0] * LATENT_CHANNELS,
        scale_factor_temporal=4,
        scale_factor_spatial=8,
    )


def build_tiny_text_encoder(tokenizer: PreTrainedTokenizerBase) -> UMT5EncoderModel:
    """Build a tiny UMT5 encoder whose hidden size matches the tiny Wan transformers."""
    return UMT5EncoderModel(
        UMT5Config(
            vocab_size=len(tokenizer),
            d_model=TEXT_DIM,
            d_ff=64,
            d_kv=16,
            num_heads=2,
            num_layers=1,
            num_decoder_layers=1,
            dropout_rate=0.0,
            is_encoder_decoder=True,
        )
    )


def save_pipeline(
    pipeline: WanPipeline,
    output_dir: Path,
    overwrite: bool,
    recipe: TinyWanRecipe,
    repo_id: str = "REPLACE_WITH_REPO_ID",
) -> None:
    """Save the tiny pipeline in Diffusers format."""
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Re-run with --overwrite to replace it.")
        if not output_dir.is_dir():
            raise NotADirectoryError(f"{output_dir} exists but is not a directory.")
        remove_existing_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    pipeline.save_pretrained(output_dir, safe_serialization=True)
    patch_fastvideo_runtime_weights(output_dir=output_dir, recipe=recipe)
    refresh_weight_component_directories(output_dir=output_dir, recipe=recipe)
    patch_saved_pipeline_metadata(output_dir=output_dir, recipe=recipe)
    write_model_card(output_dir=output_dir, recipe=recipe, repo_id=repo_id)


def remove_existing_output_dir(output_dir: Path) -> None:
    """Remove a generated output directory, retrying transient filesystem races."""
    last_error: OSError | None = None
    for _ in range(3):
        try:
            shutil.rmtree(output_dir)
            return
        except OSError as error:
            last_error = error
            time.sleep(0.2)

    assert last_error is not None
    raise last_error


def patch_saved_pipeline_metadata(output_dir: Path, recipe: TinyWanRecipe) -> None:
    """Patch saved Diffusers metadata for recipe-specific runtime compatibility."""
    model_index_path = output_dir / "model_index.json"
    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    model_index["_class_name"] = recipe.model_index_class_name

    if recipe.model_index_class_name != "WanPipeline" or recipe.fastvideo_backend is not None:
        model_index.pop("boundary_ratio", None)
        model_index.pop("expand_timesteps", None)
        for component_name in list(model_index):
            if model_index[component_name] == [None, None]:
                model_index.pop(component_name)

    model_index_path.write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")


def patch_fastvideo_runtime_weights(output_dir: Path, recipe: TinyWanRecipe) -> None:
    """Add FastVideo runtime-only weights that are not emitted by Diffusers."""
    if recipe.fastvideo_backend != "VIDEO_SPARSE_ATTN":
        return

    transformer_weights_path = single_safetensors_path(output_dir / "transformer")
    state_dict = load_safetensors_file(str(transformer_weights_path))
    dtype = state_dict["blocks.0.attn1.to_q.weight"].dtype
    hidden_size = recipe.transformer_num_attention_heads * recipe.transformer_attention_head_dim

    for block_index in range(recipe.transformer_num_layers):
        weight_name = f"blocks.{block_index}.to_gate_compress.weight"
        bias_name = f"blocks.{block_index}.to_gate_compress.bias"
        state_dict.setdefault(weight_name, torch.zeros((hidden_size, hidden_size), dtype=dtype))
        state_dict.setdefault(bias_name, torch.zeros((hidden_size,), dtype=dtype))

    save_safetensors_file(state_dict, str(transformer_weights_path), metadata={"format": "pt"})


def single_safetensors_path(component_dir: Path) -> Path:
    """Return the single safetensors file for a generated component."""
    paths = sorted(component_dir.glob("*.safetensors"))
    if not paths and component_dir.is_dir():
        marker_path = component_dir / ".dir-refresh"
        marker_path.write_text("", encoding="utf-8")
        marker_path.unlink()
        paths = sorted(component_dir.glob("*.safetensors"))
    if not paths:
        for filename in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
            direct_path = component_dir / filename
            if direct_path.is_file():
                return direct_path
    if len(paths) != 1:
        raise AssertionError(f"Expected exactly one safetensors file in {component_dir}, found {len(paths)}.")
    return paths[0]


def verify_saved_pipeline(output_dir: Path, recipe: TinyWanRecipe) -> str:
    """Verify the saved artifact with the cheapest recipe-appropriate check."""
    if recipe.diffusers_smoke_test:
        shape = run_smoke_test(output_dir)
        return f"Diffusers smoke test output shape: {shape}"

    verify_fastvideo_metadata(output_dir=output_dir, recipe=recipe)
    return "FastVideo metadata check passed"


def verify_fastvideo_metadata(output_dir: Path, recipe: TinyWanRecipe) -> None:
    """Validate metadata required for FastVideo registry and component loading."""
    model_index = json.loads((output_dir / "model_index.json").read_text(encoding="utf-8"))
    expected_components = {
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    }
    missing_components = expected_components - model_index.keys()
    if missing_components:
        missing_text = ", ".join(sorted(missing_components))
        raise AssertionError(f"FastVideo artifact is missing model_index components: {missing_text}.")

    if model_index["_class_name"] != recipe.model_index_class_name:
        raise AssertionError(
            f"Expected model_index _class_name {recipe.model_index_class_name!r}, "
            f"got {model_index['_class_name']!r}."
        )
    if model_index["transformer"] != ["diffusers", "WanTransformer3DModel"]:
        raise AssertionError(f"Unexpected transformer entry: {model_index['transformer']!r}.")
    if "transformer_2" in model_index:
        raise AssertionError("FastVideo single-transformer artifact should not advertise transformer_2.")

    verify_fastvideo_attention_backend_config(output_dir=output_dir, recipe=recipe)
    verify_required_weight_files(output_dir=output_dir, component_names=required_weight_component_names(recipe))


def verify_fastvideo_attention_backend_config(output_dir: Path, recipe: TinyWanRecipe) -> None:
    """Validate recipe-specific FastVideo backend constraints."""
    if recipe.fastvideo_backend != "VIDEO_SPARSE_ATTN":
        return

    transformer_config = json.loads((output_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    attention_head_dim = transformer_config["attention_head_dim"]
    if attention_head_dim not in {64, 128}:
        raise AssertionError(
            "FastVideo VIDEO_SPARSE_ATTN requires attention_head_dim 64 or 128, "
            f"got {attention_head_dim}."
        )
    verify_vsa_gate_compress_weights(output_dir=output_dir, recipe=recipe)


def verify_vsa_gate_compress_weights(output_dir: Path, recipe: TinyWanRecipe) -> None:
    """Verify VSA-specific FastVideo transformer weights are present."""
    transformer_weights_path = single_safetensors_path(output_dir / "transformer")
    state_dict = load_safetensors_file(str(transformer_weights_path))
    hidden_size = recipe.transformer_num_attention_heads * recipe.transformer_attention_head_dim

    for block_index in range(recipe.transformer_num_layers):
        weight_name = f"blocks.{block_index}.to_gate_compress.weight"
        bias_name = f"blocks.{block_index}.to_gate_compress.bias"
        if weight_name not in state_dict or bias_name not in state_dict:
            raise AssertionError(f"VSA artifact is missing {weight_name} or {bias_name}.")
        if tuple(state_dict[weight_name].shape) != (hidden_size, hidden_size):
            raise AssertionError(f"Unexpected {weight_name} shape: {tuple(state_dict[weight_name].shape)}.")
        if tuple(state_dict[bias_name].shape) != (hidden_size,):
            raise AssertionError(f"Unexpected {bias_name} shape: {tuple(state_dict[bias_name].shape)}.")



def required_weight_component_names(recipe: TinyWanRecipe) -> tuple[str, ...]:
    """Return model components that must have visible safetensors weights."""
    component_names = ["text_encoder", "vae", "transformer"]
    if recipe.include_transformer_2:
        component_names.append("transformer_2")
    return tuple(component_names)


def refresh_weight_component_directories(output_dir: Path, recipe: TinyWanRecipe) -> None:
    """Force fresh directory entries for saved weight components.

    Some shared filesystems can make a just-written safetensors file visible to
    direct stat/open calls before it appears in glob/listdir. FastVideo discovers
    component weights with glob, so the saved artifact must satisfy that contract.
    """
    for component_name in required_weight_component_names(recipe):
        component_dir = output_dir / component_name
        if component_dir.is_dir():
            marker_path = component_dir / ".dir-refresh"
            marker_path.write_text("", encoding="utf-8")
            marker_path.unlink()


def verify_required_weight_files(output_dir: Path, component_names: tuple[str, ...]) -> None:
    """Verify saved model components include at least one safetensors weight file."""
    missing_weights = []
    for component_name in component_names:
        component_dir = output_dir / component_name
        if not component_dir.is_dir():
            missing_weights.append(component_name)
            continue

        if not any(component_dir.glob("*.safetensors")):
            marker_path = component_dir / ".dir-refresh"
            marker_path.write_text("", encoding="utf-8")
            marker_path.unlink()

        if not any(component_dir.glob("*.safetensors")):
            missing_weights.append(component_name)

    if missing_weights:
        missing_text = ", ".join(missing_weights)
        raise AssertionError(f"Saved artifact is missing safetensors weights for: {missing_text}.")


@torch.inference_mode()
def run_smoke_test(model_path: Path) -> tuple[int, int, int, int]:
    """Reload the saved pipeline, run one denoising step, and verify video export."""
    pipeline = WanPipeline.from_pretrained(model_path)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to("cpu")

    frames = pipeline(
        prompt="debug prompt",
        height=SMOKE_HEIGHT,
        width=SMOKE_WIDTH,
        num_frames=SMOKE_FRAMES,
        num_inference_steps=SMOKE_STEPS,
        guidance_scale=1.0,
        max_sequence_length=SMOKE_MAX_SEQUENCE_LENGTH,
    ).frames[0]

    expected_shape = (SMOKE_FRAMES, SMOKE_HEIGHT, SMOKE_WIDTH, 3)
    if tuple(frames.shape) != expected_shape:
        raise AssertionError(f"Expected smoke output shape {expected_shape}, got {tuple(frames.shape)}.")

    with tempfile.TemporaryDirectory(prefix="tiny-wan-export-") as tmp_dir:
        export_path = Path(tmp_dir) / "smoke.mp4"
        export_to_video(frames, str(export_path), fps=1)
        if not export_path.is_file():
            raise AssertionError(f"Video export did not create {export_path}.")

    return expected_shape


def resolve_repo_id(repo_id: str | None, repo_name: str, token: str) -> str:
    """Resolve the target repo id from an explicit value or the token owner."""
    if repo_id:
        return repo_id

    user = HfApi(token=token).whoami()["name"]
    return f"{user}/{repo_name}"


def push_folder(output_dir: Path, repo_id: str, token: str) -> None:
    """Upload the saved Diffusers pipeline folder to the Hugging Face Hub."""
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add tiny Wan debug pipeline",
    )


def write_model_card(output_dir: Path, recipe: TinyWanRecipe, repo_id: str) -> None:
    """Write the public model card that travels with the generated HF artifact."""
    loading_example = f"""from diffusers import WanPipeline

pipe = WanPipeline.from_pretrained("{repo_id}")
pipe.set_progress_bar_config(disable=True)
frames = pipe(
    prompt="debug prompt",
    height=64,
    width=64,
    num_frames=5,
    num_inference_steps=1,
    guidance_scale=1.0,
    max_sequence_length=8,
).frames[0]"""
    if recipe.fastvideo_backend is not None:
        backend = recipe.fastvideo_backend or "TORCH_SDPA"
        generator_kwargs = [
            f'    "{repo_id}",',
            "    num_gpus=1,",
        ]
        if recipe.fastvideo_flow_shift is not None:
            generator_kwargs.append(f'    pipeline_config={{"flow_shift": {recipe.fastvideo_flow_shift}}},')
        if recipe.fastvideo_vsa_sparsity is not None:
            generator_kwargs.append(f"    VSA_sparsity={recipe.fastvideo_vsa_sparsity},")

        generate_kwargs = [
            '        "debug prompt",',
            '        output_path="my_videos/",',
            "        save_video=True,",
            "        height=64,",
            "        width=64,",
            "        num_frames=5,",
            "        num_inference_steps=3,",
            "        guidance_scale=1.0,",
        ]

        loading_example = f"""import os
from fastvideo import VideoGenerator

os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "{backend}"
generator = VideoGenerator.from_pretrained(
{chr(10).join(generator_kwargs)}
)
try:
    generator.generate_video(
{chr(10).join(generate_kwargs)}
    )
finally:
    generator.shutdown()"""

    card = f"""---
license: apache-2.0
library_name: diffusers
pipeline_tag: text-to-video
tags:
- diffusers
- wan
- tiny-random
- debug
---

# {recipe.model_card_title}

This is a randomly initialized, tiny `{recipe.model_index_class_name}` fixture for `{recipe.source_model_id}`.
{recipe.source_description}

It is intended for fast load-path and inference-control debugging only. It is not trained and should
not be used for generation quality evaluation.

```python
{loading_example}
```
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def require_hf_token() -> str:
    """Return HF_TOKEN or fail with a direct error before attempting a push."""
    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError("HF_TOKEN is required when --push is set.")
    return token
