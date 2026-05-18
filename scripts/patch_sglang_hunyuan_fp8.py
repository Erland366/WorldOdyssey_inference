#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
VENDORED_MODEL_OPT_FP8 = ROOT_DIR / "patches" / "sglang_hunyuan_fp8" / "modelopt_fp8.py"
QUANTIZATION_INIT = Path(
    "sglang/multimodal_gen/runtime/layers/quantization/__init__.py"
)
QUANTIZATION_MODEL_OPT_FP8 = Path(
    "sglang/multimodal_gen/runtime/layers/quantization/modelopt_fp8.py"
)
LINEAR_LAYER = Path("sglang/multimodal_gen/runtime/layers/linear.py")
LAYERNORM_LAYER = Path("sglang/multimodal_gen/runtime/layers/layernorm.py")
ROTARY_UTILS = Path("sglang/multimodal_gen/runtime/layers/rotary_embedding/utils.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch sglang==0.5.10.post1 for HunyuanVideo ModelOpt FP8 diffusion."
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=ROOT_DIR / ".venv_sglang",
        help="SGLang FP8 virtual environment to patch.",
    )
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=None,
        help="Patch this site-packages directory directly; used by tests.",
    )
    return parser.parse_args()


def find_site_packages(venv_path: Path) -> Path:
    candidates = sorted((venv_path / "lib").glob("python*/site-packages"))
    matching = [path for path in candidates if (path / QUANTIZATION_INIT).exists()]
    if not matching:
        raise SystemExit(f"Could not find SGLang site-packages under {venv_path}.")
    if len(matching) > 1:
        joined = ", ".join(str(path) for path in matching)
        raise SystemExit(f"Found multiple SGLang site-packages directories: {joined}")
    return matching[0]


def replace_once(text: str, old: str, new: str, description: str) -> str:
    if old not in text:
        raise SystemExit(f"Patch anchor not found for {description}.")
    return text.replace(old, new, 1)


def install_modelopt_fp8(site_packages: Path) -> None:
    target = site_packages / QUANTIZATION_MODEL_OPT_FP8
    target.parent.mkdir(parents=True, exist_ok=True)
    source = VENDORED_MODEL_OPT_FP8.read_text(encoding="utf-8")
    if target.exists() and target.read_text(encoding="utf-8") == source:
        print(f"modelopt_fp8.py already installed at {target}")
        return
    target.write_text(source, encoding="utf-8")
    print(f"installed {target}")


def patch_quantization_registry(site_packages: Path) -> None:
    path = site_packages / QUANTIZATION_INIT
    text = path.read_text(encoding="utf-8")
    original = text

    import_block = (
        "from sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8 import (\n"
        "    ModelOptFp8Config as ModelOptFp8DiffusionConfig,\n"
        ")\n"
    )
    if import_block not in text:
        fp8_import = (
            "from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config\n"
        )
        text = replace_once(
            text,
            fp8_import,
            fp8_import + import_block,
            "ModelOpt FP8 import registration",
        )

    if 'Literal["modelopt", "fp8", "modelopt_fp4", "modelslim"]' not in text:
        text = replace_once(
            text,
            'Literal["fp8", "modelopt_fp4", "modelslim"]',
            'Literal["modelopt", "fp8", "modelopt_fp4", "modelslim"]',
            "ModelOpt FP8 Literal registration",
        )

    if '"modelopt": ModelOptFp8DiffusionConfig,' not in text:
        text = replace_once(
            text,
            "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {\n",
            '_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {\n    "modelopt": ModelOptFp8DiffusionConfig,\n',
            "ModelOpt FP8 config map registration",
        )

    if text == original:
        print(f"quantization registry already patched at {path}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"patched quantization registry at {path}")


def patch_linear_scale_loader(site_packages: Path) -> None:
    path = site_packages / LINEAR_LAYER
    text = path.read_text(encoding="utf-8")
    patch = (
        '        if (\n'
        '            type(param).__name__ == "PerTensorScaleParameter"\n'
        "            and param.numel() == 1\n"
        "            and loaded_weight.ndim == 1\n"
        "            and loaded_weight.numel() > 1\n"
        "        ):\n"
        "            loaded_weight = loaded_weight.max().reshape_as(param)\n\n"
    )
    if patch in text:
        print(f"linear scale loader already patched at {path}")
        return

    anchor = (
        "        if len(loaded_weight.shape) == 0:\n"
        "            loaded_weight = loaded_weight.reshape(1)\n\n"
    )
    text = replace_once(
        text,
        anchor,
        anchor + patch,
        "fused PerTensorScaleParameter scale loader",
    )
    path.write_text(text, encoding="utf-8")
    print(f"patched linear scale loader at {path}")


def patch_layernorm_deterministic_fallback(site_packages: Path) -> None:
    path = site_packages / LAYERNORM_LAYER
    text = path.read_text(encoding="utf-8")
    original = text
    fallback = (
        '        if get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_INFERENCE"):\n'
        "            self._forward_method = self.forward_native\n\n"
    )

    scale_residual_anchor = (
        '        else:\n'
        '            raise NotImplementedError(f"Norm type {self.norm_type} not implemented")\n\n'
        "    def forward_cuda(\n"
        "        self,\n"
        "        residual: torch.Tensor,\n"
    )
    scale_residual_patch = (
        '        else:\n'
        '            raise NotImplementedError(f"Norm type {self.norm_type} not implemented")\n\n'
        + fallback
        + "    def forward_cuda(\n"
        "        self,\n"
        "        residual: torch.Tensor,\n"
    )
    if scale_residual_patch not in text:
        text = replace_once(
            text,
            scale_residual_anchor,
            scale_residual_patch,
            "deterministic fallback for fused residual norm scale shift",
        )

    norm_scale_anchor = (
        '        else:\n'
        '            raise NotImplementedError(f"Norm type {self.norm_type} not implemented")\n\n'
        "    def forward_cuda(\n"
        "        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor\n"
    )
    norm_scale_patch = (
        '        else:\n'
        '            raise NotImplementedError(f"Norm type {self.norm_type} not implemented")\n\n'
        + fallback
        + "    def forward_cuda(\n"
        "        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor\n"
    )
    if norm_scale_patch not in text:
        text = replace_once(
            text,
            norm_scale_anchor,
            norm_scale_patch,
            "deterministic fallback for fused norm scale shift",
        )

    if text == original:
        print(f"layernorm deterministic fallback already patched at {path}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"patched layernorm deterministic fallback at {path}")


def patch_flashinfer_rope_env_guard(site_packages: Path) -> None:
    path = site_packages / ROTARY_UTILS
    text = path.read_text(encoding="utf-8")
    original = text

    import_anchor = "from typing import Optional, Tuple\n\nimport torch\n"
    import_patch = "from typing import Optional, Tuple\n\nimport os\n\nimport torch\n"
    if import_patch not in text:
        text = replace_once(
            text,
            import_anchor,
            import_patch,
            "SGLANG_DISABLE_FLASHINFER_ROPE import",
        )

    env_guard = (
        "\n"
        "def _env_enabled(name: str) -> bool:\n"
        '    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}\n'
        "\n"
        "\n"
    )
    platform_anchor = "_is_cuda = current_platform.is_cuda()\n"
    if env_guard not in text:
        text = replace_once(
            text,
            platform_anchor,
            env_guard + platform_anchor,
            "SGLANG_DISABLE_FLASHINFER_ROPE helper",
        )

    flashinfer_anchor = "if _is_cuda:\n"
    flashinfer_patch = 'if _is_cuda and not _env_enabled("SGLANG_DISABLE_FLASHINFER_ROPE"):\n'
    if flashinfer_patch not in text:
        text = replace_once(
            text,
            flashinfer_anchor,
            flashinfer_patch,
            "SGLANG_DISABLE_FLASHINFER_ROPE guard",
        )

    if text == original:
        print(f"FlashInfer RoPE env guard already patched at {path}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"patched FlashInfer RoPE env guard at {path}")


def main() -> None:
    args = parse_args()
    site_packages = args.site_packages if args.site_packages is not None else find_site_packages(args.venv)
    if not (site_packages / QUANTIZATION_INIT).exists():
        raise SystemExit(f"SGLang quantization registry not found under {site_packages}.")
    if not (site_packages / LINEAR_LAYER).exists():
        raise SystemExit(f"SGLang linear layer not found under {site_packages}.")
    if not (site_packages / LAYERNORM_LAYER).exists():
        raise SystemExit(f"SGLang layernorm layer not found under {site_packages}.")
    if not (site_packages / ROTARY_UTILS).exists():
        raise SystemExit(f"SGLang rotary embedding utils not found under {site_packages}.")

    install_modelopt_fp8(site_packages)
    patch_quantization_registry(site_packages)
    patch_linear_scale_loader(site_packages)
    patch_layernorm_deterministic_fallback(site_packages)
    patch_flashinfer_rope_env_guard(site_packages)


if __name__ == "__main__":
    main()
