from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from worldodyssey_inference.video_backend.models import (
    ProviderCapability,
    VideoGenerationRequest,
    VideoJobRecord,
    VideoMode,
)
from worldodyssey_inference.video_backend.storage import JobPaths


DEFAULT_SGLANG_MODEL = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
DEFAULT_SGLANG_HEIGHT = 448
DEFAULT_SGLANG_WIDTH = 832
DEFAULT_SGLANG_NUM_FRAMES = 61
DEFAULT_SGLANG_NUM_INFERENCE_STEPS = 3
DEFAULT_SGLANG_SEED = 123
DEFAULT_SGLANG_ATTENTION_BACKEND = "video_sparse_attn"
DEFAULT_SGLANG_VSA_SPARSITY = 0.5


class ProviderError(RuntimeError):
    code = "provider_error"
    retryable = False


class UnsupportedRequestError(ProviderError):
    code = "unsupported_request"


class ProviderUnavailableError(ProviderError):
    code = "provider_unavailable"


class ProviderRuntimeError(ProviderError):
    code = "provider_runtime_error"


@dataclass(frozen=True)
class ProviderRunResult:
    output_path: Path
    metrics: dict[str, float | int | str]


class VideoProvider(Protocol):
    capability: ProviderCapability

    def validate_request(self, request: VideoGenerationRequest) -> None:
        ...

    def run(self, record: VideoJobRecord, paths: JobPaths) -> ProviderRunResult:
        ...


class DisabledProvider:
    def __init__(self, capability: ProviderCapability, message: str) -> None:
        self.capability = capability
        self._message = message

    def validate_request(self, request: VideoGenerationRequest) -> None:
        raise ProviderUnavailableError(self._message)

    def run(self, record: VideoJobRecord, paths: JobPaths) -> ProviderRunResult:
        raise ProviderUnavailableError(self._message)


class LocalSGLangProvider:
    def __init__(
        self,
        *,
        repo_root: Path,
        venv_path: Path | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.venv_path = venv_path or repo_root / ".venv_sglangcuda12"
        self.cuda_home = self.venv_path / "lib" / "python3.12" / "site-packages" / "nvidia"
        self.capability = ProviderCapability(
            id="sglang",
            label="Local SGLang Diffusion",
            enabled=True,
            local=True,
            models=[DEFAULT_SGLANG_MODEL],
            modes=[VideoMode.TEXT_TO_VIDEO],
            supports_audio=False,
            supports_seed=True,
            supports_custom_resolution=True,
            resolutions=["custom"],
            setup={
                "venv": str(self.venv_path),
                "cuda_home": str(self.cuda_home),
                "script": "scripts/install_sglang_diffusion.sh",
            },
        )

    def validate_request(self, request: VideoGenerationRequest) -> None:
        if request.mode != VideoMode.TEXT_TO_VIDEO:
            raise UnsupportedRequestError("Local SGLang currently supports text_to_video only.")
        if request.model not in self.capability.models:
            valid = ", ".join(self.capability.models)
            raise UnsupportedRequestError(f"Unsupported SGLang model {request.model!r}. Valid models: {valid}.")
        if request.image_url or request.image_base64 or request.end_image_url or request.reference_image_urls:
            raise UnsupportedRequestError("Local SGLang text_to_video does not accept image inputs yet.")
        if request.video_url:
            raise UnsupportedRequestError("Local SGLang text_to_video does not accept video inputs.")
        if request.options.duration is not None and request.options.num_frames is None:
            raise UnsupportedRequestError("Local SGLang needs options.num_frames instead of duration.")
        if request.options.resolution is not None and (
            request.options.width is None or request.options.height is None
        ):
            raise UnsupportedRequestError("Local SGLang needs explicit width and height instead of resolution only.")
        missing = [
            field
            for field in ("height", "width", "num_frames")
            if getattr(request.options, field) is None
        ]
        if missing:
            joined = ", ".join(f"options.{field}" for field in missing)
            raise UnsupportedRequestError(f"Local SGLang requires explicit {joined}.")
        if request.options.attention_backend != DEFAULT_SGLANG_ATTENTION_BACKEND:
            raise UnsupportedRequestError(
                f"Local SGLang FastWan VSA requires options.attention_backend={DEFAULT_SGLANG_ATTENTION_BACKEND!r}."
            )
        if request.options.vsa_sparsity is None:
            raise UnsupportedRequestError("Local SGLang FastWan VSA requires explicit options.vsa_sparsity.")
        if request.options.generate_audio:
            raise UnsupportedRequestError("Local SGLang FastWan does not generate audio.")
        self._validate_runtime_paths()

    def build_command(self, request: VideoGenerationRequest, output_path: Path) -> list[str]:
        options = request.options
        command = [
            "sglang",
            "generate",
            "--model-path",
            request.model,
            f"--attention-backend={options.attention_backend or DEFAULT_SGLANG_ATTENTION_BACKEND}",
            f"--VSA-sparsity={options.vsa_sparsity if options.vsa_sparsity is not None else DEFAULT_SGLANG_VSA_SPARSITY}",
            f"--num-gpus={options.num_gpus}",
            "--prompt",
            request.prompt,
            f"--height={options.height or DEFAULT_SGLANG_HEIGHT}",
            f"--width={options.width or DEFAULT_SGLANG_WIDTH}",
            f"--num-frames={options.num_frames or DEFAULT_SGLANG_NUM_FRAMES}",
            f"--num-inference-steps={options.num_inference_steps or DEFAULT_SGLANG_NUM_INFERENCE_STEPS}",
            f"--seed={options.seed if options.seed is not None else DEFAULT_SGLANG_SEED}",
            "--save-output",
            "--output-path",
            str(output_path.parent),
            "--output-file-name",
            output_path.name,
            f"--log-level={options.log_level}",
        ]
        if request.negative_prompt:
            command.extend(["--negative-prompt", request.negative_prompt])
        if options.guidance_scale is not None:
            command.append(f"--guidance-scale={options.guidance_scale}")
        if options.fps is not None:
            command.append(f"--fps={options.fps}")
        return command

    def runtime_env(self) -> dict[str, str]:
        self._validate_runtime_paths()
        env = dict(os.environ)
        env["PATH"] = f"{self.venv_path / 'bin'}:/usr/local/bin:/usr/bin:/bin"
        env["CC"] = "/usr/bin/gcc"
        env["CXX"] = "/usr/bin/g++"
        env["CUDA_HOME"] = str(self.cuda_home)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def run(self, record: VideoJobRecord, paths: JobPaths) -> ProviderRunResult:
        request = record.request
        self.validate_request(request)
        paths.output_path.parent.mkdir(parents=True, exist_ok=True)
        paths.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = self.build_command(request, paths.output_path)

        started = time.perf_counter()
        with paths.log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("$ " + " ".join(command) + "\n\n")
            log_handle.flush()
            try:
                result = subprocess.run(
                    command,
                    cwd=self.repo_root,
                    env=self.runtime_env(),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=request.options.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ProviderRuntimeError(
                    f"SGLang generation timed out after {request.options.timeout_seconds} seconds."
                ) from exc

        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            raise ProviderRuntimeError(f"SGLang exited with code {result.returncode}. See {paths.log_path}.")
        self._validate_output(paths.output_path)
        return ProviderRunResult(
            output_path=paths.output_path,
            metrics={
                "elapsed_seconds": round(elapsed, 4),
                "returncode": result.returncode,
            },
        )

    def _validate_runtime_paths(self) -> None:
        sglang_bin = self.venv_path / "bin" / "sglang"
        libcudart = self.cuda_home / "cuda_runtime" / "lib" / "libcudart.so.12"
        if not sglang_bin.exists():
            raise ProviderUnavailableError(
                f"SGLang CLI not found at {sglang_bin}. Run scripts/install_sglang_diffusion.sh."
            )
        if not libcudart.exists():
            raise ProviderUnavailableError(
                f"CUDA runtime not found at {libcudart}. Run scripts/install_sglang_diffusion.sh."
            )

    @staticmethod
    def _validate_output(path: Path) -> None:
        if not path.exists() or path.stat().st_size == 0:
            raise ProviderRuntimeError(f"SGLang did not create a non-empty output video at {path}.")
        header = path.read_bytes()[:64]
        if not 0 <= header.find(b"ftyp") <= 32:
            raise ProviderRuntimeError(f"SGLang output does not look like an MP4 file: {path}.")


def default_providers(repo_root: Path) -> dict[str, VideoProvider]:
    return {
        "sglang": LocalSGLangProvider(repo_root=repo_root),
        "fal": DisabledProvider(
            ProviderCapability(
                id="fal",
                label="fal.ai",
                enabled=False,
                local=False,
                models=["bytedance/seedance-2.0/image-to-video"],
                modes=[VideoMode.IMAGE_TO_VIDEO],
                supports_audio=True,
                supports_seed=True,
                supports_custom_resolution=False,
                supports_end_image=True,
                resolutions=["480p", "720p", "1080p"],
                setup={"env": ["FAL_KEY"]},
            ),
            "fal.ai adapter is not implemented in this local server yet.",
        ),
        "google_veo": DisabledProvider(
            ProviderCapability(
                id="google_veo",
                label="Google Veo",
                enabled=False,
                local=False,
                models=["veo-3.1-generate-preview"],
                modes=[
                    VideoMode.TEXT_TO_VIDEO,
                    VideoMode.IMAGE_TO_VIDEO,
                    VideoMode.REFERENCE_TO_VIDEO,
                    VideoMode.EXTEND_VIDEO,
                ],
                supports_audio=True,
                supports_seed=False,
                supports_custom_resolution=False,
                supports_reference_images=True,
                supports_end_image=True,
                resolutions=["720p", "1080p", "4k"],
                setup={"env": ["GEMINI_API_KEY"]},
            ),
            "Google Veo adapter is not implemented in this local server yet.",
        ),
        "xai_grok": DisabledProvider(
            ProviderCapability(
                id="xai_grok",
                label="xAI Grok Imagine",
                enabled=False,
                local=False,
                models=["grok-imagine-video"],
                modes=[
                    VideoMode.TEXT_TO_VIDEO,
                    VideoMode.IMAGE_TO_VIDEO,
                    VideoMode.REFERENCE_TO_VIDEO,
                    VideoMode.EDIT_VIDEO,
                    VideoMode.EXTEND_VIDEO,
                ],
                supports_audio=True,
                supports_seed=False,
                supports_custom_resolution=False,
                supports_reference_images=True,
                resolutions=["480p", "720p"],
                setup={"env": ["XAI_API_KEY"]},
            ),
            "xAI Grok Imagine adapter is not implemented in this local server yet.",
        ),
    }
