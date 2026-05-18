from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_sglang_hunyuan_fp8_patch_is_idempotent(tmp_path: Path) -> None:
    site_packages = tmp_path / "site-packages"
    quantization_dir = site_packages / "sglang/multimodal_gen/runtime/layers/quantization"
    layers_dir = site_packages / "sglang/multimodal_gen/runtime/layers"
    rotary_dir = layers_dir / "rotary_embedding"
    quantization_dir.mkdir(parents=True)
    layers_dir.mkdir(parents=True, exist_ok=True)
    rotary_dir.mkdir(parents=True, exist_ok=True)

    quantization_init = quantization_dir / "__init__.py"
    quantization_init.write_text(
        "\n".join(
            [
                "from typing import Literal, get_args",
                "from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (",
                "    QuantizationConfig,",
                ")",
                "from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config",
                "from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (",
                "    ModelOptFp4Config,",
                ")",
                "from sglang.multimodal_gen.runtime.layers.quantization.modelslim import ModelSlimConfig",
                "",
                'QuantizationMethods = Literal["fp8", "modelopt_fp4", "modelslim"]',
                "",
                "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {",
                '    "modelopt_fp4": ModelOptFp4Config,',
                '    "modelslim": ModelSlimConfig,',
                '    "fp8": Fp8Config,',
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    linear_py = layers_dir / "linear.py"
    linear_py.write_text(
        "\n".join(
            [
                "class LinearBase:",
                "    def weight_loader(self, param, loaded_weight):",
                "        if len(loaded_weight.shape) == 0:",
                "            loaded_weight = loaded_weight.reshape(1)",
                "",
                "        assert param.size() == loaded_weight.size(), (",
                "            f'Tried to load weights of size {loaded_weight.size()}'",
                "            f'to a parameter of size {param.size()}'",
                "        )",
                "        param.data.copy_(loaded_weight)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    layernorm_py = layers_dir / "layernorm.py"
    layernorm_py.write_text(
        "\n".join(
            [
                "class _ScaleResidualNormScaleShift:",
                "    def __init__(self):",
                "        if self.norm_type == 'rms':",
                "            self.norm = object()",
                "        elif self.norm_type == 'layer':",
                "            self.norm = object()",
                "        else:",
                "            raise NotImplementedError(f\"Norm type {self.norm_type} not implemented\")",
                "",
                "    def forward_cuda(",
                "        self,",
                "        residual: torch.Tensor,",
                "    ):",
                "        pass",
                "",
                "class _NormScaleShift:",
                "    def __init__(self):",
                "        if self.norm_type == 'rms':",
                "            self.norm = object()",
                "        elif self.norm_type == 'layer':",
                "            self.norm = object()",
                "        else:",
                "            raise NotImplementedError(f\"Norm type {self.norm_type} not implemented\")",
                "",
                "    def forward_cuda(",
                "        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor",
                "    ):",
                "        pass",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rotary_utils_py = rotary_dir / "utils.py"
    rotary_utils_py.write_text(
        "\n".join(
            [
                '"""Primitive RoPE ops."""',
                "",
                "from typing import Optional, Tuple",
                "",
                "import torch",
                "",
                "_is_cuda = current_platform.is_cuda()",
                "if _is_cuda:",
                "    try:",
                "        from flashinfer.rope import (",
                "            apply_rope_with_cos_sin_cache_inplace as _flashinfer_apply_rope_inplace,",
                "        )",
                "    except Exception:",
                "        _flashinfer_apply_rope_inplace = None",
                "else:",
                "    _flashinfer_apply_rope_inplace = None",
                "",
            ]
        ),
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "patch_sglang_hunyuan_fp8.py"
    first = subprocess.run(
        [sys.executable, str(script), "--site-packages", str(site_packages)],
        check=True,
        text=True,
        capture_output=True,
    )
    patched_registry = quantization_init.read_text(encoding="utf-8")
    patched_linear = linear_py.read_text(encoding="utf-8")
    patched_layernorm = layernorm_py.read_text(encoding="utf-8")
    patched_rotary_utils = rotary_utils_py.read_text(encoding="utf-8")

    second = subprocess.run(
        [sys.executable, str(script), "--site-packages", str(site_packages)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert (quantization_dir / "modelopt_fp8.py").exists()
    assert "installed" in first.stdout
    assert "already" in second.stdout
    assert quantization_init.read_text(encoding="utf-8") == patched_registry
    assert linear_py.read_text(encoding="utf-8") == patched_linear
    assert layernorm_py.read_text(encoding="utf-8") == patched_layernorm
    assert rotary_utils_py.read_text(encoding="utf-8") == patched_rotary_utils
    assert 'Literal["modelopt", "fp8", "modelopt_fp4", "modelslim"]' in patched_registry
    assert '"modelopt": ModelOptFp8DiffusionConfig,' in patched_registry
    assert 'type(param).__name__ == "PerTensorScaleParameter"' in patched_linear
    assert patched_layernorm.count("SGLANG_ENABLE_DETERMINISTIC_INFERENCE") == 2
    assert 'import os' in patched_rotary_utils
    assert 'def _env_enabled(name: str) -> bool:' in patched_rotary_utils
    assert 'if _is_cuda and not _env_enabled("SGLANG_DISABLE_FLASHINFER_ROPE"):' in patched_rotary_utils


def test_modelopt_fp8_patch_repairs_offloaded_weight_layout() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "patches"
        / "sglang_hunyuan_fp8"
        / "modelopt_fp8.py"
    ).read_text(encoding="utf-8")

    assert "def _ensure_column_major_weight" in source
    assert "weight.stride(0) == 1" in source
    assert "weight.t().contiguous().t()" in source
    assert "layer.weight.data = weight" in source


def test_memory_offload_launcher_uses_hunyuan_fp8_low_memory_defaults() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "serve_sglang_diffusion.sh"
    ).read_text(encoding="utf-8")

    assert 'USE_TRITON_W8A8_FP8_KERNEL=1' in source
    assert 'SGLANG_DISABLE_FLASHINFER_ROPE=1' in source
    assert 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' in source
    assert 'WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_MIN_HEIGHT:-128' in source
    assert 'WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_STRIDE_HEIGHT:-96' in source
    assert 'WORLDODYSSEY_SGLANG_VAE_TILE_SAMPLE_MIN_NUM_FRAMES:-8' in source
    assert '--vae-config.tile-sample-min-height "$VAE_TILE_SAMPLE_MIN_HEIGHT"' in source
    assert (
        '--vae-config.tile-sample-stride-num-frames "$VAE_TILE_SAMPLE_STRIDE_NUM_FRAMES"'
        in source
    )
