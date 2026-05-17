from __future__ import annotations

import base64
import binascii
import json
import os
import shutil
import time
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
    ) -> None:
        self.repo_root = repo_root
        self.venv_path = venv_path or repo_root / ".venv_sglangcuda12"
        self.cuda_home = self.venv_path / "lib" / "python3.12" / "site-packages" / "nvidia"
        self.server_url = (server_url or os.environ.get(SGLANG_SERVER_URL_ENV) or DEFAULT_SGLANG_SERVER_URL).rstrip("/")
        self.server_model = server_model or os.environ.get(SGLANG_SERVER_MODEL_ENV)
        self.capability = ProviderCapability(
            id="sglang",
            label="Local SGLang Diffusion",
            enabled=True,
            local=True,
            models=[],
            modes=[VideoMode.TEXT_TO_VIDEO, VideoMode.IMAGE_TO_VIDEO],
            supports_audio=False,
            supports_seed=False,
            supports_custom_resolution=True,
            supports_reference_images=False,
            resolutions=["custom"],
            setup={
                "venv": str(self.venv_path),
                "cuda_home": str(self.cuda_home),
                "install_script": "scripts/install_sglang_diffusion.sh",
                "server_script": "scripts/serve_sglang_diffusion.sh",
                "server_url": self.server_url,
                "server_api": "/v1/videos",
                "server_url_env": SGLANG_SERVER_URL_ENV,
                "server_model_env": SGLANG_SERVER_MODEL_ENV,
                "configured_server_model_hint": self.server_model,
                "model_policy": "request_model_forwarded_to_native_sglang",
                "default_text_to_video_model": DEFAULT_SGLANG_MODEL,
                "default_image_to_video_model": DEFAULT_SGLANG_I2V_MODEL,
                "tiny_debug_model": DEBUG_TINY_WAN_T2V_MODEL,
            },
        )

    def validate_request(self, request: VideoGenerationRequest) -> None:
        if request.mode == VideoMode.TEXT_TO_VIDEO:
            self._validate_t2v_request(request)
        elif request.mode == VideoMode.IMAGE_TO_VIDEO:
            self._validate_i2v_request(request)
        else:
            raise UnsupportedRequestError("Local SGLang currently supports text_to_video and image_to_video only.")
        self._validate_native_server_request_options(request)
        self._validate_runtime_paths()

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
        if request.options.generate_audio:
            raise UnsupportedRequestError("Local SGLang text_to_video does not generate audio.")

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
        if request.options.generate_audio:
            raise UnsupportedRequestError("Local SGLang I2V does not generate audio.")

    @staticmethod
    def _validate_native_server_request_options(request: VideoGenerationRequest) -> None:
        options = request.options
        unsupported_fields = []
        if options.num_inference_steps is not None:
            unsupported_fields.append("options.num_inference_steps")
        if options.seed is not None:
            unsupported_fields.append("options.seed")
        if options.guidance_scale is not None:
            unsupported_fields.append("options.guidance_scale")
        if options.attention_backend is not None:
            unsupported_fields.append("options.attention_backend")
        if options.vsa_sparsity is not None:
            unsupported_fields.append("options.vsa_sparsity")
        if options.num_gpus != 1:
            unsupported_fields.append("options.num_gpus")
        if options.provider_options:
            unsupported_fields.append("options.provider_options")
        if unsupported_fields:
            joined = ", ".join(unsupported_fields)
            raise UnsupportedRequestError(
                "Native SGLang serve does not accept these as per-request fields: "
                f"{joined}. Configure model, GPUs, denoising, VSA, and offload settings when starting "
                "scripts/serve_sglang_diffusion.sh."
            )

    def build_server_payload(
        self,
        request: VideoGenerationRequest,
        *,
        image_path: Path | None = None,
    ) -> dict[str, Any]:
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
        return payload

    def run(self, record: VideoJobRecord, paths: JobPaths) -> ProviderRunResult:
        request = record.request
        self.validate_request(request)
        paths.output_path.parent.mkdir(parents=True, exist_ok=True)
        paths.log_path.parent.mkdir(parents=True, exist_ok=True)
        image_path = self._stage_i2v_image(request, paths) if request.mode == VideoMode.IMAGE_TO_VIDEO else None
        payload = self.build_server_payload(request, image_path=image_path)

        started = time.perf_counter()
        with paths.log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"SGLang server: {self.server_url}\n")
            log_handle.write("POST /v1/videos\n")
            log_handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n\n")
            log_handle.flush()

            created = self._post_json("/v1/videos", payload, timeout=request.options.timeout_seconds)
            log_handle.write("Create response:\n")
            log_handle.write(json.dumps(created, indent=2, sort_keys=True) + "\n\n")
            log_handle.flush()

            video_id = self._require_string(created, "id")
            completed = self._wait_for_video(video_id, request.options.timeout_seconds, log_handle)
            status = self._require_string(completed, "status")
            if status != "completed":
                error = completed.get("error")
                raise ProviderRuntimeError(f"SGLang video job {video_id} finished with status {status!r}: {error}")

            self._download_video(video_id, paths.output_path, timeout=request.options.timeout_seconds)
            log_handle.write(f"Downloaded /v1/videos/{video_id}/content to {paths.output_path}\n")

        elapsed = time.perf_counter() - started
        self._validate_output(paths.output_path)
        return ProviderRunResult(
            output_path=paths.output_path,
            metrics={
                "elapsed_seconds": round(elapsed, 4),
                "sglang_video_id": video_id,
                "sglang_server_url": self.server_url,
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

    def _wait_for_video(self, video_id: str, timeout_seconds: int, log_handle) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        path = f"/v1/videos/{quote(video_id, safe='')}"
        while True:
            status_payload = self._get_json(path, timeout=timeout_seconds)
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

    def _post_json(self, path: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            self._url(path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._open_json(request, timeout=timeout)

    def _get_json(self, path: str, *, timeout: int) -> dict[str, Any]:
        request = Request(self._url(path), method="GET")
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
                f"SGLang server is not reachable at {self.server_url}. Start scripts/serve_sglang_diffusion.sh first: {exc}"
            ) from exc

        try:
            decoded = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ProviderRuntimeError(f"SGLang server returned invalid JSON for {request.full_url}.") from exc
        if not isinstance(decoded, dict):
            raise ProviderRuntimeError(f"SGLang server returned non-object JSON for {request.full_url}.")
        return decoded

    def _download_video(self, video_id: str, output_path: Path, *, timeout: int) -> None:
        path = f"/v1/videos/{quote(video_id, safe='')}/content"
        request = Request(self._url(path), method="GET")
        try:
            with urlopen(request, timeout=timeout) as response, output_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderRuntimeError(f"SGLang server returned HTTP {exc.code} for {request.full_url}: {detail}") from exc
        except URLError as exc:
            raise ProviderUnavailableError(f"SGLang server is not reachable at {self.server_url}: {exc}") from exc

    def _url(self, path: str) -> str:
        return f"{self.server_url}/{path.lstrip('/')}"

    @staticmethod
    def _require_string(payload: dict[str, Any], field: str) -> str:
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise ProviderRuntimeError(f"SGLang response is missing string field {field!r}.")
        return value

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

    @staticmethod
    def _stage_i2v_url(image_url: str, paths: JobPaths) -> Path:
        parsed = urlparse(image_url)
        if parsed.scheme not in {"http", "https"}:
            raise UnsupportedRequestError("image_url must use http or https.")
        suffix = Path(parsed.path).suffix
        if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".png"
        target = paths.job_dir / f"input_image{suffix}"
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
