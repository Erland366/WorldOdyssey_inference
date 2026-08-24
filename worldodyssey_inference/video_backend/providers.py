from __future__ import annotations

import base64
import binascii
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from worldodyssey_inference.video_backend.models import (
    DEFAULT_IMAGE_TO_VIDEO_MODEL,
    DEFAULT_TEXT_TO_VIDEO_MODEL,
    ProviderCapability,
    VideoGenerationRequest,
    VideoJobRecord,
    VideoMode,
)
from worldodyssey_inference.video_backend.storage import JobPaths


DEFAULT_SGLANG_MODEL = DEFAULT_TEXT_TO_VIDEO_MODEL
DEFAULT_SGLANG_I2V_MODEL = DEFAULT_IMAGE_TO_VIDEO_MODEL
DEBUG_TINY_WAN_T2V_MODEL = "Erland/tiny-wan2.1-t2v-debug"
DEFAULT_SGLANG_HEIGHT = 448
DEFAULT_SGLANG_WIDTH = 832
DEFAULT_SGLANG_NUM_FRAMES = 61
DEFAULT_SGLANG_I2V_HEIGHT = 480
DEFAULT_SGLANG_I2V_WIDTH = 832
DEFAULT_SGLANG_I2V_NUM_FRAMES = 81
DEFAULT_SGLANG_FPS = 16
DEFAULT_SGLANG_SERVER_URL = "http://127.0.0.1:30000"
SGLANG_SERVER_URL_ENV = "WORLDODYSSEY_SGLANG_BASE_URL"
SGLANG_SERVER_MODEL_ENV = "WORLDODYSSEY_SGLANG_MODEL"
SGLANG_SERVER_VIDEO_API_FORMAT_ENV = "WORLDODYSSEY_SGLANG_VIDEO_API_FORMAT"
MINIMAX_H3_MODEL = "MiniMaxAI/MiniMax-H3"
MINIMAX_H3_FL2VA_SERVER_URL_ENV = "WORLDODYSSEY_MINIMAX_H3_FL2VA_BASE_URL"
MINIMAX_H3_REF2VA_SERVER_URL_ENV = "WORLDODYSSEY_MINIMAX_H3_REF2VA_BASE_URL"
DEFAULT_MINIMAX_H3_FL2VA_SERVER_URL = "http://127.0.0.1:30010"
DEFAULT_MINIMAX_H3_REF2VA_SERVER_URL = "http://127.0.0.1:30011"
MINIMAX_H3_PROVIDER_OPTIONS_KEY = "minimax_h3"
MINIMAX_H3_PROVIDER_OPTION_KEYS = frozenset(
    {
        "audio_flow_shift",
        "flow_shift",
        "frame_index",
        "num_outputs_per_prompt",
        "quality",
        "short_edge",
    }
)
MINIMAX_H3_FPS = 24
MINIMAX_H3_DEFAULT_DURATION_SECONDS = 5
MINIMAX_H3_MIN_DURATION_SECONDS = 4
MINIMAX_H3_MAX_DURATION_SECONDS = 15
MINIMAX_H3_DEFAULT_SHORT_EDGE = 768
MINIMAX_H3_DEFAULT_INFERENCE_STEPS = 50
MINIMAX_H3_DEFAULT_FLOW_SHIFT = 12.0
MINIMAX_H3_DEFAULT_AUDIO_FLOW_SHIFT = 3.0
MINIMAX_H3_ASPECT_RATIOS = frozenset(
    {"auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
)
SGLANG_VIDEO_API_FORMAT_JSON = "json"
SGLANG_VIDEO_API_FORMAT_MULTIPART = "multipart"
DEFAULT_SGLANG_VIDEO_API_FORMAT = SGLANG_VIDEO_API_FORMAT_MULTIPART
SGLANG_VIDEO_API_FORMATS = frozenset(
    {SGLANG_VIDEO_API_FORMAT_JSON, SGLANG_VIDEO_API_FORMAT_MULTIPART}
)
SGLANG_MULTIPART_PROVIDER_OPTION_KEYS = frozenset({"request_fields", "extra_body", "extra_params"})
SGLANG_MANAGED_VIDEO_FIELDS = frozenset(
    {
        "prompt",
        "model",
        "size",
        "fps",
        "num_frames",
        "seconds",
        "input_reference",
        "negative_prompt",
        "num_inference_steps",
        "seed",
        "guidance_scale",
        "generate_sound",
    }
)


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
        server_url: str | None = None,
        server_model: str | None = None,
        server_api_format: str | None = None,
        minimax_h3_fl2va_server_url: str | None = None,
        minimax_h3_ref2va_server_url: str | None = None,
        minimax_h3_venv_path: Path | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.venv_path = venv_path or repo_root / ".venv_sglang"
        self.cuda_home = self.venv_path / "lib" / "python3.12" / "site-packages" / "nvidia"
        self.server_url = (server_url or os.environ.get(SGLANG_SERVER_URL_ENV) or DEFAULT_SGLANG_SERVER_URL).rstrip("/")
        self.server_model = server_model or os.environ.get(SGLANG_SERVER_MODEL_ENV)
        self.minimax_h3_venv_path = minimax_h3_venv_path or repo_root / ".venv_sglang_h3"
        self.minimax_h3_fl2va_server_url = (
            minimax_h3_fl2va_server_url
            or os.environ.get(MINIMAX_H3_FL2VA_SERVER_URL_ENV)
            or DEFAULT_MINIMAX_H3_FL2VA_SERVER_URL
        ).rstrip("/")
        self.minimax_h3_ref2va_server_url = (
            minimax_h3_ref2va_server_url
            or os.environ.get(MINIMAX_H3_REF2VA_SERVER_URL_ENV)
            or DEFAULT_MINIMAX_H3_REF2VA_SERVER_URL
        ).rstrip("/")
        self.server_api_format = self._normalize_server_api_format(
            server_api_format
            or os.environ.get(SGLANG_SERVER_VIDEO_API_FORMAT_ENV)
            or DEFAULT_SGLANG_VIDEO_API_FORMAT
        )
        self.capability = ProviderCapability(
            id="sglang",
            label="Local SGLang Diffusion",
            enabled=True,
            local=True,
            models=[],
            modes=[VideoMode.TEXT_TO_VIDEO, VideoMode.IMAGE_TO_VIDEO, VideoMode.REFERENCE_TO_VIDEO],
            supports_audio=True,
            supports_seed=True,
            supports_custom_resolution=True,
            supports_reference_images=True,
            supports_end_image=True,
            resolutions=["custom"],
            setup={
                "venv": str(self.venv_path),
                "cuda_home": str(self.cuda_home),
                "install_script": "scripts/install_sglang_diffusion.sh",
                "server_script": "scripts/serve_sglang_diffusion.sh",
                "server_url": self.server_url,
                "server_api": "/v1/videos",
                "server_api_format": self.server_api_format,
                "server_url_env": SGLANG_SERVER_URL_ENV,
                "server_model_env": SGLANG_SERVER_MODEL_ENV,
                "server_api_format_env": SGLANG_SERVER_VIDEO_API_FORMAT_ENV,
                "configured_server_model_hint": self.server_model,
                "model_policy": "request_model_forwarded_to_native_sglang",
                "default_text_to_video_model": DEFAULT_SGLANG_MODEL,
                "default_image_to_video_model": DEFAULT_SGLANG_I2V_MODEL,
                "tiny_debug_model": DEBUG_TINY_WAN_T2V_MODEL,
                "minimax_h3_model": MINIMAX_H3_MODEL,
                "minimax_h3_fl2va_server_url": self.minimax_h3_fl2va_server_url,
                "minimax_h3_ref2va_server_url": self.minimax_h3_ref2va_server_url,
                "minimax_h3_fl2va_server_url_env": MINIMAX_H3_FL2VA_SERVER_URL_ENV,
                "minimax_h3_ref2va_server_url_env": MINIMAX_H3_REF2VA_SERVER_URL_ENV,
            },
        )

    def validate_request(self, request: VideoGenerationRequest) -> None:
        if self._is_minimax_h3_request(request):
            self._validate_minimax_h3_request(request)
            self._validate_runtime_paths(self.minimax_h3_venv_path)
            return
        if request.mode == VideoMode.TEXT_TO_VIDEO:
            self._validate_t2v_request(request)
        elif request.mode == VideoMode.IMAGE_TO_VIDEO:
            self._validate_i2v_request(request)
        else:
            raise UnsupportedRequestError("Local SGLang currently supports text_to_video and image_to_video only.")
        self._validate_native_server_request_options(request)
        self._validate_runtime_paths()

    def _validate_minimax_h3_request(self, request: VideoGenerationRequest) -> None:
        if request.mode not in {
            VideoMode.TEXT_TO_VIDEO,
            VideoMode.IMAGE_TO_VIDEO,
            VideoMode.REFERENCE_TO_VIDEO,
        }:
            raise UnsupportedRequestError(
                "MiniMax-H3 supports text_to_video, image_to_video, and reference_to_video."
            )
        if request.negative_prompt is not None:
            raise UnsupportedRequestError("MiniMax-H3 does not accept negative_prompt.")
        if request.video_url:
            raise UnsupportedRequestError(
                "This MiniMax-H3 integration currently supports text and image conditions only."
            )
        if request.options.guidance_scale is not None:
            raise UnsupportedRequestError(
                "MiniMax-H3 uses a CFG-distilled checkpoint and does not accept guidance_scale."
            )
        if request.options.generate_audio is False:
            raise UnsupportedRequestError(
                "MiniMax-H3 jointly generates video and audio; generate_audio cannot be false."
            )
        if request.options.fps is not None and request.options.fps != MINIMAX_H3_FPS:
            raise UnsupportedRequestError(f"MiniMax-H3 supports {MINIMAX_H3_FPS} FPS only.")
        if (
            request.options.width is not None
            or request.options.height is not None
            or request.options.resolution is not None
        ):
            raise UnsupportedRequestError(
                "MiniMax-H3 uses a fixed 768-pixel short edge; set aspect_ratio "
                "instead of width, height, or resolution."
            )
        aspect_ratio = request.options.aspect_ratio or "auto"
        if aspect_ratio not in MINIMAX_H3_ASPECT_RATIOS:
            raise UnsupportedRequestError(
                f"Unsupported MiniMax-H3 aspect_ratio {aspect_ratio!r}."
            )
        if request.options.num_gpus != 1:
            raise UnsupportedRequestError(
                "options.num_gpus is a server-launch setting for MiniMax-H3, not a request field."
            )
        if request.options.attention_backend is not None or request.options.vsa_sparsity is not None:
            raise UnsupportedRequestError(
                "MiniMax-H3 attention configuration belongs on the model-server launcher."
            )

        duration = self._minimax_h3_duration_seconds(request)
        if not MINIMAX_H3_MIN_DURATION_SECONDS <= duration <= MINIMAX_H3_MAX_DURATION_SECONDS:
            raise UnsupportedRequestError(
                "MiniMax-H3 duration must be between "
                f"{MINIMAX_H3_MIN_DURATION_SECONDS} and {MINIMAX_H3_MAX_DURATION_SECONDS} seconds."
            )

        primary_count = sum(
            bool(value)
            for value in (
                request.image_path,
                request.image_url,
                request.image_base64,
            )
        )
        if primary_count > 1:
            raise UnsupportedRequestError(
                "MiniMax-H3 accepts at most one of image_path, image_url, or image_base64 as the primary image."
            )
        if request.mode == VideoMode.TEXT_TO_VIDEO:
            if primary_count or request.end_image_url or request.reference_image_urls:
                raise UnsupportedRequestError("MiniMax-H3 text_to_video accepts a text prompt only.")
        elif request.mode == VideoMode.IMAGE_TO_VIDEO:
            if primary_count != 1:
                raise UnsupportedRequestError(
                    "MiniMax-H3 image_to_video requires exactly one primary image."
                )
            if request.reference_image_urls:
                raise UnsupportedRequestError(
                    "Use reference_to_video for MiniMax-H3 semantic reference images."
                )
            if aspect_ratio != "auto":
                raise UnsupportedRequestError(
                    "MiniMax-H3 FL2VA requires aspect_ratio='auto' so keyframe geometry is preserved."
                )
        else:
            if request.end_image_url:
                raise UnsupportedRequestError(
                    "MiniMax-H3 reference_to_video does not accept an endpoint image."
                )
            reference_count = primary_count + len(request.reference_image_urls)
            if not 1 <= reference_count <= 9:
                raise UnsupportedRequestError(
                    "MiniMax-H3 reference_to_video requires between one and nine reference images."
                )

        self._validate_minimax_h3_provider_options(request.options.provider_options)

    def _validate_t2v_request(self, request: VideoGenerationRequest) -> None:
        if request.image_path or request.image_url or request.image_base64 or request.end_image_url or request.reference_image_urls:
            raise UnsupportedRequestError("Local SGLang text_to_video does not accept image inputs.")
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

    def _validate_i2v_request(self, request: VideoGenerationRequest) -> None:
        image_inputs = [
            bool(request.image_path),
            bool(request.image_url),
            bool(request.image_base64),
        ]
        if sum(image_inputs) != 1:
            raise UnsupportedRequestError(
                "Local SGLang image_to_video requires exactly one of image_path, image_url, or image_base64."
            )
        if request.end_image_url or request.reference_image_urls:
            raise UnsupportedRequestError("Local SGLang image_to_video currently accepts one start image only.")
        if request.video_url:
            raise UnsupportedRequestError("Local SGLang image_to_video does not accept video inputs.")
        if request.options.duration is not None and request.options.num_frames is None:
            raise UnsupportedRequestError("Local SGLang needs options.num_frames instead of duration.")
        if request.options.resolution is not None and (
            request.options.width is None or request.options.height is None
        ):
            raise UnsupportedRequestError("Local SGLang needs explicit width and height instead of resolution only.")

    def _validate_native_server_request_options(self, request: VideoGenerationRequest) -> None:
        options = request.options
        unsupported_fields = []
        if options.attention_backend is not None:
            unsupported_fields.append("options.attention_backend")
        if options.vsa_sparsity is not None:
            unsupported_fields.append("options.vsa_sparsity")
        if options.num_gpus != 1:
            unsupported_fields.append("options.num_gpus")
        if self.server_api_format == SGLANG_VIDEO_API_FORMAT_JSON:
            if request.negative_prompt is not None:
                unsupported_fields.append("negative_prompt")
            if options.num_inference_steps is not None:
                unsupported_fields.append("options.num_inference_steps")
            if options.seed is not None:
                unsupported_fields.append("options.seed")
            if options.guidance_scale is not None:
                unsupported_fields.append("options.guidance_scale")
            if options.generate_audio is not None:
                unsupported_fields.append("options.generate_audio")
            if options.provider_options:
                unsupported_fields.append("options.provider_options")
        if unsupported_fields:
            joined = ", ".join(unsupported_fields)
            raise UnsupportedRequestError(
                "Native SGLang serve does not accept these as per-request fields: "
                f"{joined}. Configure model, GPUs, VSA, and offload settings when starting "
                "scripts/serve_sglang_diffusion.sh. For newer SGLang servers that accept multipart "
                f"request fields, set {SGLANG_SERVER_VIDEO_API_FORMAT_ENV}=multipart."
            )
        if self.server_api_format == SGLANG_VIDEO_API_FORMAT_MULTIPART:
            self._validate_multipart_provider_options(options.provider_options)

    @staticmethod
    def _validate_minimax_h3_provider_options(provider_options: dict[str, Any]) -> None:
        unknown_containers = sorted(set(provider_options) - {MINIMAX_H3_PROVIDER_OPTIONS_KEY})
        if unknown_containers:
            joined = ", ".join(f"options.provider_options.{key}" for key in unknown_containers)
            raise UnsupportedRequestError(
                f"MiniMax-H3 provider options must be nested under {MINIMAX_H3_PROVIDER_OPTIONS_KEY!r}. "
                f"Unsupported: {joined}."
            )
        options = provider_options.get(MINIMAX_H3_PROVIDER_OPTIONS_KEY, {})
        if not isinstance(options, dict):
            raise UnsupportedRequestError(
                f"options.provider_options.{MINIMAX_H3_PROVIDER_OPTIONS_KEY} must be an object."
            )
        unknown_keys = sorted(set(options) - MINIMAX_H3_PROVIDER_OPTION_KEYS)
        if unknown_keys:
            joined = ", ".join(
                f"options.provider_options.{MINIMAX_H3_PROVIDER_OPTIONS_KEY}.{key}"
                for key in unknown_keys
            )
            raise UnsupportedRequestError(f"Unsupported MiniMax-H3 request options: {joined}.")

        frame_index = options.get("frame_index", 0)
        if frame_index not in {0, -1} or isinstance(frame_index, bool):
            raise UnsupportedRequestError("MiniMax-H3 frame_index must be 0 (first) or -1 (last).")
        for name in ("short_edge", "num_outputs_per_prompt"):
            value = options.get(name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
            ):
                raise UnsupportedRequestError(f"MiniMax-H3 {name} must be a positive integer.")
        if options.get("short_edge", MINIMAX_H3_DEFAULT_SHORT_EDGE) != MINIMAX_H3_DEFAULT_SHORT_EDGE:
            raise UnsupportedRequestError(
                f"MiniMax-H3 short_edge must be {MINIMAX_H3_DEFAULT_SHORT_EDGE}."
            )
        for name in ("flow_shift", "audio_flow_shift"):
            value = options.get(name)
            if value is not None and (
                not isinstance(value, int | float) or isinstance(value, bool) or value <= 0
            ):
                raise UnsupportedRequestError(f"MiniMax-H3 {name} must be a positive number.")
        quality = options.get("quality")
        if quality is not None and quality not in {"lossless", "high"}:
            raise UnsupportedRequestError("MiniMax-H3 quality must be 'lossless' or 'high'.")

    @staticmethod
    def _is_minimax_h3_request(request: VideoGenerationRequest) -> bool:
        return request.model.rstrip("/").rsplit("/", maxsplit=1)[-1].lower() == "minimax-h3"

    @staticmethod
    def _minimax_h3_task(request: VideoGenerationRequest) -> str:
        if request.mode == VideoMode.TEXT_TO_VIDEO:
            return "t2va"
        if request.mode == VideoMode.IMAGE_TO_VIDEO:
            return "fl2va"
        return "ref2va"

    @staticmethod
    def _minimax_h3_duration_seconds(request: VideoGenerationRequest) -> float:
        options = request.options
        if options.duration is not None:
            return float(options.duration)
        if options.num_frames is not None:
            fps = options.fps or MINIMAX_H3_FPS
            return options.num_frames / fps
        return float(MINIMAX_H3_DEFAULT_DURATION_SECONDS)

    def _build_minimax_h3_payload(
        self,
        request: VideoGenerationRequest,
        *,
        image_path: Path | None,
        end_image_path: Path | None,
        reference_image_paths: list[Path],
    ) -> dict[str, Any]:
        options = request.options
        h3_options = options.provider_options.get(MINIMAX_H3_PROVIDER_OPTIONS_KEY, {})
        task = self._minimax_h3_task(request)
        duration = self._minimax_h3_duration_seconds(request)
        conditions: list[dict[str, Any]] = []

        if task == "fl2va":
            frame_index = int(h3_options.get("frame_index", 0))
            if image_path is not None:
                conditions.append(
                    {
                        "type": "image",
                        "uri": image_path.resolve().as_uri(),
                        "role": "keyframe",
                        "frame_index": frame_index,
                    }
                )
            if end_image_path is not None:
                if frame_index == -1:
                    raise UnsupportedRequestError(
                        "A last-frame primary image cannot be combined with end_image_url."
                    )
                conditions.append(
                    {
                        "type": "image",
                        "uri": end_image_path.resolve().as_uri(),
                        "role": "keyframe",
                        "frame_index": -1,
                    }
                )
        elif task == "ref2va":
            all_references = ([image_path] if image_path is not None else []) + reference_image_paths
            conditions.extend(
                {
                    "type": "image",
                    "uri": path.resolve().as_uri(),
                    "role": "reference",
                }
                for path in all_references
            )

        aspect_ratio = options.aspect_ratio or "auto"
        short_edge = int(h3_options.get("short_edge", MINIMAX_H3_DEFAULT_SHORT_EDGE))

        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "seconds": duration,
            "task": task,
            "conditions": conditions,
            "target": {
                "short_edge": short_edge,
                "aspect_ratio": aspect_ratio,
                "duration_seconds": duration,
            },
            "num_outputs_per_prompt": int(h3_options.get("num_outputs_per_prompt", 1)),
            "num_inference_steps": options.num_inference_steps or MINIMAX_H3_DEFAULT_INFERENCE_STEPS,
            "flow_shift": float(h3_options.get("flow_shift", MINIMAX_H3_DEFAULT_FLOW_SHIFT)),
            "audio_flow_shift": float(
                h3_options.get("audio_flow_shift", MINIMAX_H3_DEFAULT_AUDIO_FLOW_SHIFT)
            ),
        }
        if options.seed is not None:
            payload["seed"] = options.seed
        if "quality" in h3_options:
            payload["quality"] = h3_options["quality"]
        return payload

    def _request_api_format(self, request: VideoGenerationRequest) -> str:
        if self._is_minimax_h3_request(request):
            return SGLANG_VIDEO_API_FORMAT_JSON
        return self.server_api_format

    def _server_url_for_request(self, request: VideoGenerationRequest) -> str:
        if not self._is_minimax_h3_request(request):
            return self.server_url
        if self._minimax_h3_task(request) == "ref2va":
            return self.minimax_h3_ref2va_server_url
        return self.minimax_h3_fl2va_server_url

    def build_server_payload(
        self,
        request: VideoGenerationRequest,
        *,
        image_path: Path | None = None,
        end_image_path: Path | None = None,
        reference_image_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        if self._is_minimax_h3_request(request):
            return self._build_minimax_h3_payload(
                request,
                image_path=image_path,
                end_image_path=end_image_path,
                reference_image_paths=reference_image_paths or [],
            )
        options = request.options
        is_i2v = request.mode == VideoMode.IMAGE_TO_VIDEO
        height = options.height or (DEFAULT_SGLANG_I2V_HEIGHT if is_i2v else DEFAULT_SGLANG_HEIGHT)
        width = options.width or (DEFAULT_SGLANG_I2V_WIDTH if is_i2v else DEFAULT_SGLANG_WIDTH)
        num_frames = options.num_frames or (
            DEFAULT_SGLANG_I2V_NUM_FRAMES if is_i2v else DEFAULT_SGLANG_NUM_FRAMES
        )
        fps = options.fps or DEFAULT_SGLANG_FPS
        payload: dict[str, Any] = {
            "prompt": request.prompt,
            "model": request.model,
            "size": f"{width}x{height}",
            "fps": fps,
            "num_frames": num_frames,
        }
        if options.duration is not None:
            payload["seconds"] = options.duration
        if image_path is not None:
            payload["input_reference"] = str(image_path)
        if request.negative_prompt is not None:
            payload["negative_prompt"] = request.negative_prompt
        if options.num_inference_steps is not None:
            payload["num_inference_steps"] = options.num_inference_steps
        if options.seed is not None:
            payload["seed"] = options.seed
        if options.guidance_scale is not None:
            payload["guidance_scale"] = options.guidance_scale
        if options.generate_audio is not None:
            payload["generate_sound"] = options.generate_audio
        if self.server_api_format == SGLANG_VIDEO_API_FORMAT_MULTIPART:
            self._apply_multipart_provider_options(payload, options.provider_options)
        return payload

    def run(self, record: VideoJobRecord, paths: JobPaths) -> ProviderRunResult:
        request = record.request
        self.validate_request(request)
        paths.output_path.parent.mkdir(parents=True, exist_ok=True)
        paths.log_path.parent.mkdir(parents=True, exist_ok=True)
        image_path: Path | None = None
        end_image_path: Path | None = None
        reference_image_paths: list[Path] = []
        if self._is_minimax_h3_request(request):
            image_path, end_image_path, reference_image_paths = self._stage_minimax_h3_images(request, paths)
        elif request.mode == VideoMode.IMAGE_TO_VIDEO:
            image_path = self._stage_i2v_image(request, paths)
        payload = self.build_server_payload(
            request,
            image_path=image_path,
            end_image_path=end_image_path,
            reference_image_paths=reference_image_paths,
        )
        api_format = self._request_api_format(request)
        server_url = self._server_url_for_request(request)
        files: dict[str, Path] = {}
        if api_format == SGLANG_VIDEO_API_FORMAT_MULTIPART and image_path is not None:
            payload.pop("input_reference", None)
            files["input_reference"] = image_path

        started = time.perf_counter()
        with paths.log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"SGLang server: {server_url}\n")
            log_handle.write(f"POST /v1/videos ({api_format})\n")
            log_handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n\n")
            if files:
                log_handle.write("Multipart files:\n")
                log_handle.write(json.dumps({field: str(path) for field, path in files.items()}, indent=2) + "\n\n")
            log_handle.flush()

            created = self._post_video_create(
                payload,
                files=files,
                timeout=request.options.timeout_seconds,
                api_format=api_format,
                server_url=server_url,
            )
            log_handle.write("Create response:\n")
            log_handle.write(json.dumps(created, indent=2, sort_keys=True) + "\n\n")
            log_handle.flush()

            video_id = self._require_string(created, "id")
            completed = self._wait_for_video(
                video_id,
                request.options.timeout_seconds,
                log_handle,
                server_url=server_url,
            )
            status = self._require_string(completed, "status")
            if status != "completed":
                error = completed.get("error")
                raise ProviderRuntimeError(f"SGLang video job {video_id} finished with status {status!r}: {error}")

            self._download_video(
                video_id,
                paths.output_path,
                timeout=request.options.timeout_seconds,
                server_url=server_url,
            )
            log_handle.write(f"Downloaded /v1/videos/{video_id}/content to {paths.output_path}\n")

        elapsed = time.perf_counter() - started
        self._validate_output(paths.output_path)
        return ProviderRunResult(
            output_path=paths.output_path,
            metrics={
                "elapsed_seconds": round(elapsed, 4),
                "sglang_video_id": video_id,
                "sglang_server_url": server_url,
                "sglang_video_api_format": api_format,
            },
        )

    def _validate_runtime_paths(self, venv_path: Path | None = None) -> None:
        selected_venv = venv_path or self.venv_path
        sglang_bin = selected_venv / "bin" / "sglang"
        cuda_runtime_root = selected_venv / "lib" / "python3.12" / "site-packages" / "nvidia"
        libcudart_candidates = (
            cuda_runtime_root / "cuda_runtime" / "lib" / "libcudart.so.12",
            cuda_runtime_root / "cuda_runtime" / "lib" / "libcudart.so.13",
            cuda_runtime_root / "cuda_runtime" / "lib" / "libcudart.so.13.0",
            cuda_runtime_root / "cu13" / "lib" / "libcudart.so.13",
        )
        if not sglang_bin.exists():
            raise ProviderUnavailableError(
                f"SGLang CLI not found at {sglang_bin}. Run scripts/install_sglang_diffusion.sh."
            )
        if not any(path.exists() for path in libcudart_candidates):
            raise ProviderUnavailableError(
                f"CUDA runtime not found under {cuda_runtime_root}. Run the matching SGLang installer."
            )

    def _wait_for_video(
        self,
        video_id: str,
        timeout_seconds: int,
        log_handle,
        *,
        server_url: str,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        path = f"/v1/videos/{quote(video_id, safe='')}"
        while True:
            status_payload = self._get_json(path, timeout=timeout_seconds, server_url=server_url)
            status = self._require_string(status_payload, "status")
            log_handle.write(f"Poll {video_id}: {status}\n")
            log_handle.flush()
            if status in {"completed", "failed", "deleted"}:
                return status_payload
            if time.monotonic() >= deadline:
                raise ProviderRuntimeError(
                    f"SGLang video job {video_id} did not finish within {timeout_seconds} seconds."
                )
            time.sleep(2.0)

    def _post_video_create(
        self,
        payload: dict[str, Any],
        *,
        files: dict[str, Path],
        timeout: int,
        api_format: str,
        server_url: str,
    ) -> dict[str, Any]:
        if api_format == SGLANG_VIDEO_API_FORMAT_MULTIPART:
            return self._post_multipart(
                "/v1/videos", payload, files=files, timeout=timeout, server_url=server_url
            )
        return self._post_json("/v1/videos", payload, timeout=timeout, server_url=server_url)

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: int,
        server_url: str,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self._url(path, server_url=server_url),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._open_json(request, timeout=timeout)

    def _post_multipart(
        self,
        path: str,
        fields: dict[str, Any],
        *,
        files: dict[str, Path],
        timeout: int,
        server_url: str,
    ) -> dict[str, Any]:
        boundary = f"----WorldOdyssey{uuid.uuid4().hex}"
        body = bytearray()
        for name, value in fields.items():
            encoded_value = self._encode_multipart_field(name, value)
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            body.extend(encoded_value.encode("utf-8"))
            body.extend(b"\r\n")

        for name, file_path in files.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{file_path.name}"\r\n'
                ).encode("utf-8")
            )
            body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
            body.extend(file_path.read_bytes())
            body.extend(b"\r\n")

        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        request = Request(
            self._url(path, server_url=server_url),
            data=bytes(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
            method="POST",
        )
        return self._open_json(request, timeout=timeout)

    def _get_json(self, path: str, *, timeout: int, server_url: str) -> dict[str, Any]:
        request = Request(self._url(path, server_url=server_url), method="GET")
        return self._open_json(request, timeout=timeout)

    def _open_json(self, request: Request, *, timeout: int) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderRuntimeError(f"SGLang server returned HTTP {exc.code} for {request.full_url}: {detail}") from exc
        except URLError as exc:
            raise ProviderUnavailableError(
                f"SGLang server is not reachable at {request.full_url}: {exc}"
            ) from exc

        try:
            decoded = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ProviderRuntimeError(f"SGLang server returned invalid JSON for {request.full_url}.") from exc
        if not isinstance(decoded, dict):
            raise ProviderRuntimeError(f"SGLang server returned non-object JSON for {request.full_url}.")
        return decoded

    def _download_video(
        self,
        video_id: str,
        output_path: Path,
        *,
        timeout: int,
        server_url: str,
    ) -> None:
        path = f"/v1/videos/{quote(video_id, safe='')}/content"
        request = Request(self._url(path, server_url=server_url), method="GET")
        try:
            with urlopen(request, timeout=timeout) as response, output_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderRuntimeError(f"SGLang server returned HTTP {exc.code} for {request.full_url}: {detail}") from exc
        except URLError as exc:
            raise ProviderUnavailableError(f"SGLang server is not reachable at {server_url}: {exc}") from exc

    def _url(self, path: str, *, server_url: str | None = None) -> str:
        base_url = server_url or self.server_url
        return f"{base_url}/{path.lstrip('/')}"

    @staticmethod
    def _require_string(payload: dict[str, Any], field: str) -> str:
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise ProviderRuntimeError(f"SGLang response is missing string field {field!r}.")
        return value

    @staticmethod
    def _normalize_server_api_format(value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SGLANG_VIDEO_API_FORMATS:
            allowed = ", ".join(sorted(SGLANG_VIDEO_API_FORMATS))
            raise UnsupportedRequestError(
                f"Unsupported {SGLANG_SERVER_VIDEO_API_FORMAT_ENV}={value!r}. Expected one of: {allowed}."
            )
        return normalized

    @staticmethod
    def _validate_multipart_provider_options(provider_options: dict[str, Any]) -> None:
        if not provider_options:
            return

        unknown_keys = sorted(set(provider_options) - SGLANG_MULTIPART_PROVIDER_OPTION_KEYS)
        if unknown_keys:
            joined = ", ".join(f"options.provider_options.{key}" for key in unknown_keys)
            raise UnsupportedRequestError(
                "SGLang multipart provider_options supports only request_fields, extra_body, and extra_params. "
                f"Unsupported: {joined}."
            )

        request_fields = provider_options.get("request_fields")
        if request_fields is not None:
            if not isinstance(request_fields, dict):
                raise UnsupportedRequestError("options.provider_options.request_fields must be an object.")
            for field_name, field_value in request_fields.items():
                if not isinstance(field_name, str) or not field_name:
                    raise UnsupportedRequestError(
                        "options.provider_options.request_fields keys must be non-empty strings."
                    )
                if field_name in SGLANG_MANAGED_VIDEO_FIELDS:
                    raise UnsupportedRequestError(
                        f"options.provider_options.request_fields.{field_name} conflicts with a managed SGLang field."
                    )
                LocalSGLangProvider._validate_multipart_scalar(
                    f"options.provider_options.request_fields.{field_name}",
                    field_value,
                )

        for container_name in ("extra_body", "extra_params"):
            container = provider_options.get(container_name)
            if container is not None and not isinstance(container, str):
                json.dumps(container)

    @staticmethod
    def _validate_multipart_scalar(field_name: str, field_value: Any) -> None:
        if isinstance(field_value, bool):
            return
        if isinstance(field_value, str) and field_value:
            return
        if isinstance(field_value, int | float):
            return
        raise UnsupportedRequestError(
            f"{field_name} must be a string, number, or boolean multipart field. "
            "Use options.provider_options.extra_params for structured JSON."
        )

    @staticmethod
    def _apply_multipart_provider_options(payload: dict[str, Any], provider_options: dict[str, Any]) -> None:
        if not provider_options:
            return
        request_fields = provider_options.get("request_fields") or {}
        payload.update(request_fields)
        for container_name in ("extra_body", "extra_params"):
            if container_name in provider_options:
                container = provider_options[container_name]
                payload[container_name] = (
                    container if isinstance(container, str) else json.dumps(container, sort_keys=True)
                )

    @staticmethod
    def _encode_multipart_field(name: str, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return value
        if isinstance(value, int | float):
            return str(value)
        raise UnsupportedRequestError(f"Cannot encode SGLang multipart field {name!r} with value type {type(value).__name__}.")

    @staticmethod
    def _validate_output(path: Path) -> None:
        if not path.exists() or path.stat().st_size == 0:
            raise ProviderRuntimeError(f"SGLang did not create a non-empty output video at {path}.")
        header = path.read_bytes()[:64]
        if not 0 <= header.find(b"ftyp") <= 32:
            raise ProviderRuntimeError(f"SGLang output does not look like an MP4 file: {path}.")

    def _stage_i2v_image(self, request: VideoGenerationRequest, paths: JobPaths) -> Path:
        paths.job_dir.mkdir(parents=True, exist_ok=True)
        if request.image_path:
            image_path = Path(request.image_path).expanduser()
            if not image_path.is_absolute():
                image_path = self.repo_root / image_path
            if not image_path.exists():
                raise ProviderUnavailableError(f"Image input file not found: {image_path}")
            return image_path
        if request.image_url:
            return self._stage_i2v_url(request.image_url, paths)
        if request.image_base64:
            return self._stage_i2v_base64(request.image_base64, paths)
        raise UnsupportedRequestError("Local SGLang image_to_video requires an image input.")

    def _stage_minimax_h3_images(
        self,
        request: VideoGenerationRequest,
        paths: JobPaths,
    ) -> tuple[Path | None, Path | None, list[Path]]:
        paths.job_dir.mkdir(parents=True, exist_ok=True)
        primary_path: Path | None = None
        if request.image_path or request.image_url or request.image_base64:
            primary_path = self._stage_i2v_image(request, paths)
        end_path = (
            self._stage_image_url(request.end_image_url, paths, stem="end_image")
            if request.end_image_url
            else None
        )
        reference_paths = [
            self._stage_image_url(url, paths, stem=f"reference_image_{index}")
            for index, url in enumerate(request.reference_image_urls, start=1)
        ]
        return primary_path, end_path, reference_paths

    @staticmethod
    def _stage_i2v_url(image_url: str, paths: JobPaths) -> Path:
        return LocalSGLangProvider._stage_image_url(image_url, paths, stem="input_image")

    @staticmethod
    def _stage_image_url(image_url: str, paths: JobPaths, *, stem: str) -> Path:
        parsed = urlparse(image_url)
        if parsed.scheme not in {"http", "https"}:
            raise UnsupportedRequestError("image_url must use http or https.")
        suffix = Path(parsed.path).suffix
        if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".png"
        target = paths.job_dir / f"{stem}{suffix}"
        request = Request(image_url, headers={"User-Agent": "WorldOdyssey-inference/0.1"})
        try:
            with urlopen(request, timeout=60) as response, target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except URLError as exc:
            raise ProviderUnavailableError(f"Failed to fetch image_url {image_url!r}: {exc}") from exc
        if target.stat().st_size == 0:
            raise ProviderUnavailableError(f"Downloaded image_url {image_url!r} to an empty file.")
        return target

    @staticmethod
    def _stage_i2v_base64(image_base64: str, paths: JobPaths) -> Path:
        suffix = ".png"
        payload = image_base64
        if image_base64.startswith("data:"):
            header, separator, encoded = image_base64.partition(",")
            if not separator:
                raise UnsupportedRequestError("image_base64 data URI is missing a comma separator.")
            payload = encoded
            mime_type = header.split(";", maxsplit=1)[0].removeprefix("data:")
            suffix = {
                "image/jpeg": ".jpg",
                "image/jpg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
            }.get(mime_type, ".png")
        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except binascii.Error as exc:
            raise UnsupportedRequestError("image_base64 is not valid base64.") from exc
        target = paths.job_dir / f"input_image{suffix}"
        target.write_bytes(image_bytes)
        if target.stat().st_size == 0:
            raise UnsupportedRequestError("image_base64 decoded to an empty file.")
        return target


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
                supports_reference_images=True,
                resolutions=["480p", "720p"],
                setup={"env": ["XAI_API_KEY"]},
            ),
            "xAI Grok Imagine adapter is not implemented in this local server yet.",
        ),
    }
