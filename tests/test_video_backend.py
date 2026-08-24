from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from worldodyssey_inference.video_backend.app import create_app
from worldodyssey_inference.video_backend.manager import VideoJobManager
from worldodyssey_inference.video_backend.models import (
    DEFAULT_IMAGE_TO_VIDEO_MODEL,
    JobStatus,
    ProviderCapability,
    VideoGenerationRequest,
    VideoJobRecord,
    VideoMode,
    utc_now_iso,
)
from worldodyssey_inference.video_backend.providers import (
    DEBUG_TINY_WAN_T2V_MODEL,
    DEFAULT_SGLANG_I2V_MODEL,
    DEFAULT_SGLANG_MODEL,
    LocalSGLangProvider,
    MINIMAX_H3_MODEL,
    ProviderRunResult,
    SGLANG_VIDEO_API_FORMAT_JSON,
    SGLANG_VIDEO_API_FORMAT_MULTIPART,
    UnsupportedRequestError,
)
from worldodyssey_inference.video_backend.storage import JobPaths, JobStore


def make_sglang_runtime(root: Path) -> Path:
    venv = root / ".venv_sglang"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "sglang").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    cuda_lib = venv / "lib" / "python3.12" / "site-packages" / "nvidia" / "cuda_runtime" / "lib"
    cuda_lib.mkdir(parents=True)
    (cuda_lib / "libcudart.so.12").write_text("", encoding="utf-8")
    return venv


def make_request(**overrides: object) -> VideoGenerationRequest:
    payload = {
        "provider": "sglang",
        "model": DEFAULT_SGLANG_MODEL,
        "mode": "text_to_video",
        "prompt": "A calm ocean wave at sunrise",
        "options": {
            "height": 448,
            "width": 832,
            "num_frames": 61,
        },
    }
    payload.update(overrides)
    return VideoGenerationRequest.model_validate(payload)


def make_sglang_provider(
    root: Path,
    *,
    server_model: str = DEFAULT_SGLANG_MODEL,
    server_url: str | None = None,
    server_api_format: str | None = None,
) -> LocalSGLangProvider:
    return LocalSGLangProvider(
        repo_root=root,
        venv_path=make_sglang_runtime(root),
        server_url=server_url,
        server_model=server_model,
        server_api_format=server_api_format,
    )


def test_sglang_payload_maps_unified_request_to_native_video_api(tmp_path: Path) -> None:
    provider = LocalSGLangProvider(repo_root=tmp_path, venv_path=make_sglang_runtime(tmp_path))

    payload = provider.build_server_payload(make_request())

    assert payload == {
        "prompt": "A calm ocean wave at sunrise",
        "model": DEFAULT_SGLANG_MODEL,
        "size": "832x448",
        "fps": 16,
        "num_frames": 61,
    }


def test_sglang_provider_fails_fast_for_unsupported_modes(tmp_path: Path) -> None:
    provider = make_sglang_provider(tmp_path)
    request = make_request(mode="edit_video")

    with pytest.raises(UnsupportedRequestError, match="text_to_video and image_to_video only"):
        provider.validate_request(request)


def test_sglang_text_to_video_rejects_image_inputs(tmp_path: Path) -> None:
    provider = make_sglang_provider(tmp_path)
    request = make_request(image_path="input.png")

    with pytest.raises(UnsupportedRequestError, match="text_to_video does not accept image inputs"):
        provider.validate_request(request)


def test_sglang_provider_accepts_default_model_without_backend_model_allowlist(tmp_path: Path) -> None:
    provider = make_sglang_provider(tmp_path)
    request = make_request(options={"height": 448, "width": 832, "num_frames": 61})

    provider.validate_request(request)


def test_sglang_provider_does_not_require_backend_model_hint(tmp_path: Path) -> None:
    provider = LocalSGLangProvider(repo_root=tmp_path, venv_path=make_sglang_runtime(tmp_path))
    request = make_request(options={"height": 448, "width": 832, "num_frames": 61})

    provider.validate_request(request)


def test_sglang_legacy_json_server_rejects_sampling_request_options(tmp_path: Path) -> None:
    provider = make_sglang_provider(tmp_path, server_api_format=SGLANG_VIDEO_API_FORMAT_JSON)
    request = make_request(
        options={
            "height": 448,
            "width": 832,
            "num_frames": 61,
            "num_inference_steps": 3,
            "seed": 123,
            "attention_backend": "video_sparse_attn",
            "vsa_sparsity": 0.5,
            "num_gpus": 2,
            "provider_options": {"generation_kwargs": {"boundary_ratio": 0.875}},
        },
    )

    with pytest.raises(UnsupportedRequestError, match="Native SGLang serve does not accept"):
        provider.validate_request(request)


def test_sglang_provider_accepts_custom_model_ids_for_native_server(tmp_path: Path) -> None:
    provider = make_sglang_provider(tmp_path, server_model="example/custom-sglang-diffusion-model")
    request = make_request(
        model="example/custom-sglang-diffusion-model",
        options={
            "height": 64,
            "width": 64,
            "num_frames": 5,
        },
    )

    provider.validate_request(request)
    payload = provider.build_server_payload(request)

    assert payload["model"] == "example/custom-sglang-diffusion-model"


def test_sglang_provider_forwards_model_mismatch_to_native_server(tmp_path: Path) -> None:
    provider = LocalSGLangProvider(
        repo_root=tmp_path,
        venv_path=make_sglang_runtime(tmp_path),
        server_model="loaded/model",
    )
    request = make_request(model="different/model")

    provider.validate_request(request)
    payload = provider.build_server_payload(request)

    assert payload["model"] == "different/model"


def test_sglang_capability_does_not_publish_model_allowlist(tmp_path: Path) -> None:
    provider = make_sglang_provider(tmp_path)

    assert provider.capability.models == []
    assert provider.capability.setup["model_policy"] == "request_model_forwarded_to_native_sglang"
    assert provider.capability.setup["server_api"] == "/v1/videos"
    assert provider.capability.setup["server_api_format"] == "multipart"
    assert provider.capability.supports_seed is True
    assert provider.capability.setup["configured_server_model_hint"] == DEFAULT_SGLANG_MODEL
    assert provider.capability.setup["default_text_to_video_model"] == DEFAULT_SGLANG_MODEL
    assert provider.capability.setup["default_image_to_video_model"] == DEFAULT_SGLANG_I2V_MODEL
    assert provider.capability.setup["tiny_debug_model"] == DEBUG_TINY_WAN_T2V_MODEL


def test_request_infers_image_to_video_defaults_from_image_path() -> None:
    request = VideoGenerationRequest.model_validate(
        {
            "provider": "sglang",
            "prompt": "animate this frame",
            "image_path": "input.png",
        }
    )

    assert request.mode == VideoMode.IMAGE_TO_VIDEO
    assert request.model == DEFAULT_IMAGE_TO_VIDEO_MODEL


def test_sglang_i2v_payload_uses_easy_defaults(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"fake image")
    provider = LocalSGLangProvider(repo_root=tmp_path, venv_path=make_sglang_runtime(tmp_path))
    request = VideoGenerationRequest.model_validate(
        {
            "provider": "sglang",
            "prompt": "animate this frame",
            "image_path": str(image_path),
        }
    )

    payload = provider.build_server_payload(request, image_path=image_path)

    assert payload["model"] == DEFAULT_SGLANG_I2V_MODEL
    assert payload["size"] == "832x480"
    assert payload["num_frames"] == 81
    assert payload["fps"] == 16
    assert payload["input_reference"] == str(image_path)


def test_sglang_i2v_payload_passes_staged_image_to_native_video_api(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"fake image")
    provider = LocalSGLangProvider(repo_root=tmp_path, venv_path=make_sglang_runtime(tmp_path))
    request = VideoGenerationRequest.model_validate(
        {
            "provider": "sglang",
            "model": "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
            "mode": "image_to_video",
            "prompt": "animate this frame",
            "image_path": str(image_path),
            "options": {
                "height": 448,
                "width": 832,
                "num_frames": 61,
            },
        }
    )

    payload = provider.build_server_payload(request, image_path=image_path)

    assert payload["model"] == "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers"
    assert payload["input_reference"] == str(image_path)
    assert payload["size"] == "832x448"
    assert payload["num_frames"] == 61


def test_sglang_rejects_provider_options_for_native_server(tmp_path: Path) -> None:
    provider = make_sglang_provider(tmp_path)
    request = make_request(
        options={
            "height": 448,
            "width": 832,
            "num_frames": 61,
            "provider_options": {"unknown": True},
        },
    )

    with pytest.raises(UnsupportedRequestError, match="options.provider_options"):
        provider.validate_request(request)


def test_sglang_json_api_format_rejects_negative_prompt(tmp_path: Path) -> None:
    provider = make_sglang_provider(tmp_path, server_api_format=SGLANG_VIDEO_API_FORMAT_JSON)
    request = make_request(negative_prompt="low quality")

    with pytest.raises(UnsupportedRequestError, match="negative_prompt"):
        provider.validate_request(request)


def test_sglang_multipart_api_accepts_native_request_fields(tmp_path: Path) -> None:
    provider = make_sglang_provider(
        tmp_path,
        server_api_format=SGLANG_VIDEO_API_FORMAT_MULTIPART,
    )
    request = make_request(
        negative_prompt="low quality",
        options={
            "height": 128,
            "width": 128,
            "num_frames": 5,
            "num_inference_steps": 1,
            "seed": 123,
            "guidance_scale": 4.5,
            "generate_audio": True,
            "provider_options": {
                "request_fields": {"enable_teacache": True, "flow_shift": 10.0},
                "extra_body": {"denoise": "fast"},
                "extra_params": {
                    "guardrails": False,
                    "use_resolution_template": False,
                    "use_duration_template": False,
                },
            },
        },
    )

    provider.validate_request(request)
    payload = provider.build_server_payload(request)

    assert provider.capability.supports_seed is True
    assert provider.capability.supports_audio is True
    assert provider.capability.setup["server_api_format"] == SGLANG_VIDEO_API_FORMAT_MULTIPART
    assert payload["negative_prompt"] == "low quality"
    assert payload["num_inference_steps"] == 1
    assert payload["seed"] == 123
    assert payload["guidance_scale"] == 4.5
    assert payload["generate_sound"] is True
    assert payload["enable_teacache"] is True
    assert payload["flow_shift"] == 10.0
    assert json.loads(payload["extra_body"]) == {"denoise": "fast"}
    assert json.loads(payload["extra_params"]) == {
        "guardrails": False,
        "use_duration_template": False,
        "use_resolution_template": False,
    }


def test_sglang_multipart_rejects_launch_time_options(tmp_path: Path) -> None:
    provider = make_sglang_provider(
        tmp_path,
        server_api_format=SGLANG_VIDEO_API_FORMAT_MULTIPART,
    )
    request = make_request(
        options={
            "height": 448,
            "width": 832,
            "num_frames": 61,
            "num_gpus": 2,
            "attention_backend": "video_sparse_attn",
            "vsa_sparsity": 0.5,
        },
    )

    with pytest.raises(UnsupportedRequestError, match="options.attention_backend"):
        provider.validate_request(request)


def test_sglang_multipart_rejects_managed_provider_option_fields(tmp_path: Path) -> None:
    provider = make_sglang_provider(
        tmp_path,
        server_api_format=SGLANG_VIDEO_API_FORMAT_MULTIPART,
    )
    request = make_request(
        options={
            "height": 128,
            "width": 128,
            "num_frames": 5,
            "provider_options": {"request_fields": {"seed": 123}},
        },
    )

    with pytest.raises(UnsupportedRequestError, match="conflicts with a managed SGLang field"):
        provider.validate_request(request)


def test_sglang_rejects_vsa_sparsity_as_per_request_native_server_option(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"fake image")
    provider = make_sglang_provider(tmp_path, server_model=DEFAULT_SGLANG_I2V_MODEL)
    request = VideoGenerationRequest.model_validate(
        {
            "provider": "sglang",
            "prompt": "animate this frame",
            "image_path": str(image_path),
            "options": {"vsa_sparsity": 0.5},
        }
    )

    with pytest.raises(UnsupportedRequestError, match="options.vsa_sparsity"):
        provider.validate_request(request)


def test_sglang_i2v_payload_does_not_infer_vsa_from_model_name(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"fake image")
    provider = LocalSGLangProvider(repo_root=tmp_path, venv_path=make_sglang_runtime(tmp_path))
    request = VideoGenerationRequest.model_validate(
        {
            "provider": "sglang",
            "model": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            "prompt": "animate this frame",
            "image_path": str(image_path),
        }
    )

    payload = provider.build_server_payload(request, image_path=image_path)

    assert payload["model"] == "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    assert "attention_backend" not in payload
    assert "vsa_sparsity" not in payload


def test_sglang_i2v_stages_base64_image(tmp_path: Path) -> None:
    provider = LocalSGLangProvider(repo_root=tmp_path, venv_path=make_sglang_runtime(tmp_path))
    paths = JobStore(tmp_path / "jobs").paths_for("job")
    request = VideoGenerationRequest.model_validate(
        {
            "provider": "sglang",
            "prompt": "animate this frame",
            "image_base64": "data:image/png;base64,ZmFrZSBpbWFnZQ==",
        }
    )

    image_path = provider._stage_i2v_image(request, paths)

    assert image_path == paths.job_dir / "input_image.png"
    assert image_path.read_bytes() == b"fake image"


def test_minimax_h3_fl2va_payload_uses_primary_image_as_first_frame(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"fake image")
    provider = LocalSGLangProvider(repo_root=tmp_path, venv_path=make_sglang_runtime(tmp_path))
    request = VideoGenerationRequest.model_validate(
        {
            "provider": "sglang",
            "model": MINIMAX_H3_MODEL,
            "mode": "image_to_video",
            "prompt": "The person picks up the cup while room ambience remains synchronized.",
            "image_path": str(image_path),
            "options": {
                "duration": 5,
                "fps": 24,
                "num_inference_steps": 50,
                "seed": 101,
                "generate_audio": True,
            },
        }
    )

    payload = provider.build_server_payload(request, image_path=image_path)

    assert payload == {
        "model": MINIMAX_H3_MODEL,
        "prompt": "The person picks up the cup while room ambience remains synchronized.",
        "seconds": 5.0,
        "task": "fl2va",
        "conditions": [
            {
                "type": "image",
                "uri": image_path.resolve().as_uri(),
                "role": "keyframe",
                "frame_index": 0,
            }
        ],
        "target": {
            "short_edge": 768,
            "aspect_ratio": "auto",
            "duration_seconds": 5.0,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": 50,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": 101,
    }


def test_minimax_h3_text_request_uses_t2va_on_fl2va_server(tmp_path: Path) -> None:
    provider = LocalSGLangProvider(
        repo_root=tmp_path,
        venv_path=make_sglang_runtime(tmp_path),
        minimax_h3_fl2va_server_url="http://127.0.0.1:31010",
    )
    request = VideoGenerationRequest.model_validate(
        {
            "provider": "sglang",
            "model": MINIMAX_H3_MODEL,
            "mode": "text_to_video",
            "prompt": "A quiet workshop with synchronized tool sounds.",
            "options": {"duration": 4},
        }
    )

    payload = provider.build_server_payload(request)

    assert payload["task"] == "t2va"
    assert payload["conditions"] == []
    assert provider._server_url_for_request(request) == "http://127.0.0.1:31010"


def test_minimax_h3_ref2va_payload_uses_semantic_image_references(tmp_path: Path) -> None:
    first = tmp_path / "subject.png"
    second = tmp_path / "style.png"
    first.write_bytes(b"subject")
    second.write_bytes(b"style")
    provider = LocalSGLangProvider(
        repo_root=tmp_path,
        venv_path=make_sglang_runtime(tmp_path),
        minimax_h3_ref2va_server_url="http://127.0.0.1:31011",
    )
    request = VideoGenerationRequest.model_validate(
        {
            "provider": "sglang",
            "model": MINIMAX_H3_MODEL,
            "mode": "reference_to_video",
            "prompt": "Use <Picture 1> as the subject and <Picture 2> as the visual style.",
            "image_path": str(first),
            "reference_image_urls": ["https://example.com/style.png"],
            "options": {
                "duration": 6,
                "provider_options": {
                    "minimax_h3": {
                        "quality": "lossless",
                        "flow_shift": 11.5,
                        "audio_flow_shift": 2.5,
                    }
                },
            },
        }
    )

    payload = provider.build_server_payload(
        request,
        image_path=first,
        reference_image_paths=[second],
    )

    assert payload["task"] == "ref2va"
    assert payload["conditions"] == [
        {"type": "image", "uri": first.resolve().as_uri(), "role": "reference"},
        {"type": "image", "uri": second.resolve().as_uri(), "role": "reference"},
    ]
    assert payload["quality"] == "lossless"
    assert payload["flow_shift"] == 11.5
    assert payload["audio_flow_shift"] == 2.5
    assert provider._server_url_for_request(request) == "http://127.0.0.1:31011"


def test_minimax_h3_rejects_audio_disable_and_invalid_duration(tmp_path: Path) -> None:
    runtime = make_sglang_runtime(tmp_path)
    provider = LocalSGLangProvider(
        repo_root=tmp_path,
        venv_path=runtime,
        minimax_h3_venv_path=runtime,
    )
    request = VideoGenerationRequest.model_validate(
        {
            "provider": "sglang",
            "model": MINIMAX_H3_MODEL,
            "mode": "text_to_video",
            "prompt": "A test.",
            "options": {"duration": 3, "generate_audio": False},
        }
    )

    with pytest.raises(UnsupportedRequestError, match="generate_audio cannot be false"):
        provider.validate_request(request)


def start_fake_sglang_server():
    requests: list[dict] = []
    video_bytes = b"\x00\x00\x00\x18ftypmp42fake"

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/v1/videos":
                self.send_error(404)
                return
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            content_type = self.headers.get("Content-Type", "")
            if content_type.startswith("application/json"):
                payload = json.loads(body.decode("utf-8"))
                requests.append(payload)
            elif content_type.startswith("multipart/form-data"):
                requests.append({"content_type": content_type, "body": body})
            else:
                self.send_error(415)
                return
            self._send_json({"id": "video_1", "object": "video", "status": "queued"})

        def do_GET(self) -> None:
            if self.path == "/v1/videos/video_1":
                self._send_json({"id": "video_1", "object": "video", "status": "completed"})
                return
            if self.path == "/v1/videos/video_1/content":
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(video_bytes)))
                self.end_headers()
                self.wfile.write(video_bytes)
                return
            self.send_error(404)

        def log_message(self, format: str, *args) -> None:
            return

        def _send_json(self, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, requests


def test_sglang_provider_runs_through_native_server_api(tmp_path: Path) -> None:
    server, requests = start_fake_sglang_server()
    try:
        server_url = f"http://127.0.0.1:{server.server_port}"
        provider = LocalSGLangProvider(
            repo_root=tmp_path,
            venv_path=make_sglang_runtime(tmp_path),
            server_url=server_url,
            server_model=DEFAULT_SGLANG_MODEL,
            server_api_format=SGLANG_VIDEO_API_FORMAT_JSON,
        )
        paths = JobStore(tmp_path / "jobs").paths_for("job")
        now = utc_now_iso()
        request = make_request(options={"height": 64, "width": 64, "num_frames": 5, "timeout_seconds": 5})
        record = VideoJobRecord(
            id="job",
            status=JobStatus.RUNNING,
            provider="sglang",
            model=request.model,
            mode=request.mode,
            created_at=now,
            updated_at=now,
            request=request,
        )

        result = provider.run(record, paths)

        assert result.output_path == paths.output_path
        assert paths.output_path.read_bytes() == b"\x00\x00\x00\x18ftypmp42fake"
        assert requests == [
            {
                "prompt": "A calm ocean wave at sunrise",
                "model": DEFAULT_SGLANG_MODEL,
                "size": "64x64",
                "fps": 16,
                "num_frames": 5,
            }
        ]
        assert "POST /v1/videos" in paths.log_path.read_text(encoding="utf-8")
        assert result.metrics["sglang_video_id"] == "video_1"
    finally:
        server.shutdown()
        server.server_close()


def test_minimax_h3_run_uses_json_and_fl2va_route(tmp_path: Path) -> None:
    server, requests = start_fake_sglang_server()
    try:
        image_path = tmp_path / "input.png"
        image_path.write_bytes(b"fake image")
        runtime = make_sglang_runtime(tmp_path)
        server_url = f"http://127.0.0.1:{server.server_port}"
        provider = LocalSGLangProvider(
            repo_root=tmp_path,
            venv_path=runtime,
            minimax_h3_venv_path=runtime,
            minimax_h3_fl2va_server_url=server_url,
            server_api_format=SGLANG_VIDEO_API_FORMAT_MULTIPART,
        )
        paths = JobStore(tmp_path / "jobs").paths_for("job")
        now = utc_now_iso()
        request = VideoGenerationRequest.model_validate(
            {
                "provider": "sglang",
                "model": MINIMAX_H3_MODEL,
                "mode": "image_to_video",
                "prompt": "Animate the supplied first frame with synchronized audio.",
                "image_path": str(image_path),
                "options": {"duration": 5, "seed": 101, "timeout_seconds": 5},
            }
        )
        record = VideoJobRecord(
            id="job",
            status=JobStatus.RUNNING,
            provider="sglang",
            model=request.model,
            mode=request.mode,
            created_at=now,
            updated_at=now,
            request=request,
        )

        result = provider.run(record, paths)

        assert requests[0]["task"] == "fl2va"
        assert requests[0]["conditions"] == [
            {
                "type": "image",
                "uri": image_path.resolve().as_uri(),
                "role": "keyframe",
                "frame_index": 0,
            }
        ]
        assert result.metrics["sglang_server_url"] == server_url
        assert result.metrics["sglang_video_api_format"] == "json"
        assert "POST /v1/videos (json)" in paths.log_path.read_text(encoding="utf-8")
    finally:
        server.shutdown()
        server.server_close()


def test_sglang_provider_runs_multipart_native_server_api(tmp_path: Path) -> None:
    server, requests = start_fake_sglang_server()
    try:
        server_url = f"http://127.0.0.1:{server.server_port}"
        provider = LocalSGLangProvider(
            repo_root=tmp_path,
            venv_path=make_sglang_runtime(tmp_path),
            server_url=server_url,
            server_model=DEFAULT_SGLANG_MODEL,
            server_api_format=SGLANG_VIDEO_API_FORMAT_MULTIPART,
        )
        paths = JobStore(tmp_path / "jobs").paths_for("job")
        now = utc_now_iso()
        request = make_request(
            negative_prompt="low quality",
            options={
                "height": 128,
                "width": 128,
                "num_frames": 5,
                "timeout_seconds": 5,
                "num_inference_steps": 1,
                "seed": 123,
                "guidance_scale": 4.5,
                "generate_audio": False,
                "provider_options": {
                    "request_fields": {"enable_teacache": True, "flow_shift": 10.0},
                    "extra_params": {"guardrails": False},
                },
            },
        )
        record = VideoJobRecord(
            id="job",
            status=JobStatus.RUNNING,
            provider="sglang",
            model=request.model,
            mode=request.mode,
            created_at=now,
            updated_at=now,
            request=request,
        )

        result = provider.run(record, paths)

        assert result.output_path == paths.output_path
        assert requests[0]["content_type"].startswith("multipart/form-data")
        body = requests[0]["body"]
        assert b'name="prompt"\r\n\r\nA calm ocean wave at sunrise' in body
        assert b'name="negative_prompt"\r\n\r\nlow quality' in body
        assert b'name="num_inference_steps"\r\n\r\n1' in body
        assert b'name="seed"\r\n\r\n123' in body
        assert b'name="guidance_scale"\r\n\r\n4.5' in body
        assert b'name="generate_sound"\r\n\r\nfalse' in body
        assert b'name="enable_teacache"\r\n\r\ntrue' in body
        assert b'name="flow_shift"\r\n\r\n10.0' in body
        assert b'name="extra_params"\r\n\r\n{"guardrails": false}' in body
        assert "POST /v1/videos (multipart)" in paths.log_path.read_text(encoding="utf-8")
        assert result.metrics["sglang_video_api_format"] == "multipart"
    finally:
        server.shutdown()
        server.server_close()


def test_sglang_provider_uploads_i2v_image_for_multipart_api(tmp_path: Path) -> None:
    server, requests = start_fake_sglang_server()
    try:
        image_path = tmp_path / "input.png"
        image_path.write_bytes(b"fake image")
        server_url = f"http://127.0.0.1:{server.server_port}"
        provider = LocalSGLangProvider(
            repo_root=tmp_path,
            venv_path=make_sglang_runtime(tmp_path),
            server_url=server_url,
            server_model=DEFAULT_SGLANG_I2V_MODEL,
            server_api_format=SGLANG_VIDEO_API_FORMAT_MULTIPART,
        )
        paths = JobStore(tmp_path / "jobs").paths_for("job")
        now = utc_now_iso()
        request = VideoGenerationRequest.model_validate(
            {
                "provider": "sglang",
                "model": DEFAULT_SGLANG_I2V_MODEL,
                "mode": "image_to_video",
                "prompt": "animate this frame",
                "image_path": str(image_path),
                "options": {"height": 128, "width": 128, "num_frames": 5, "timeout_seconds": 5},
            }
        )
        record = VideoJobRecord(
            id="job",
            status=JobStatus.RUNNING,
            provider="sglang",
            model=request.model,
            mode=request.mode,
            created_at=now,
            updated_at=now,
            request=request,
        )

        provider.run(record, paths)

        body = requests[0]["body"]
        assert b'name="input_reference"; filename="input.png"' in body
        assert b"fake image" in body
        assert str(image_path).encode("utf-8") not in body
    finally:
        server.shutdown()
        server.server_close()


class FakeProvider:
    capability = ProviderCapability(
        id="fake",
        label="Fake Provider",
        enabled=True,
        local=True,
        models=["fake-model"],
        modes=[VideoMode.TEXT_TO_VIDEO],
        supports_seed=True,
        supports_custom_resolution=True,
    )

    def validate_request(self, request: VideoGenerationRequest) -> None:
        if request.model != "fake-model":
            raise ValueError("bad model")

    def run(self, record, paths: JobPaths) -> ProviderRunResult:
        paths.output_path.parent.mkdir(parents=True, exist_ok=True)
        paths.log_path.write_text("fake provider ran\n", encoding="utf-8")
        paths.output_path.write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
        return ProviderRunResult(output_path=paths.output_path, metrics={"elapsed_seconds": 0.01})


def wait_for_status(client: TestClient, job_id: str, status: str, *, timeout_s: float = 5.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(f"/v1/video/generations/{job_id}")
        response.raise_for_status()
        payload = response.json()
        if payload["status"] == status:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not reach status {status!r}")


def wait_for_batch_status(client: TestClient, batch_id: str, status: str, *, timeout_s: float = 5.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(f"/v1/video/generation-batches/{batch_id}")
        response.raise_for_status()
        payload = response.json()
        if payload["status"] == status:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Batch {batch_id} did not reach status {status!r}")


def test_video_backend_submits_polls_and_serves_video(tmp_path: Path) -> None:
    manager = VideoJobManager(
        store=JobStore(tmp_path / "jobs"),
        providers={"fake": FakeProvider()},
    )
    app = create_app(repo_root=tmp_path, manager=manager)
    client = TestClient(app)

    response = client.post(
        "/v1/video/generations",
        json={
            "provider": "fake",
            "model": "fake-model",
            "mode": "text_to_video",
            "prompt": "test prompt",
            "options": {"height": 64, "width": 64, "num_frames": 5},
        },
    )

    assert response.status_code == 202
    job_id = response.json()["id"]
    record = wait_for_status(client, job_id, JobStatus.SUCCEEDED.value)
    assert record["output"]["video_url"] == f"/v1/video/generations/{job_id}/video"
    assert record["metrics"]["elapsed_seconds"] == 0.01

    video_response = client.get(f"/v1/video/generations/{job_id}/video")
    assert video_response.status_code == 200
    assert b"ftyp" in video_response.content[:32]

    logs_response = client.get(f"/v1/video/generations/{job_id}/logs")
    assert logs_response.text == "fake provider ran\n"


def test_video_backend_submits_batch_and_reports_jobs(tmp_path: Path) -> None:
    manager = VideoJobManager(
        store=JobStore(tmp_path / "jobs"),
        providers={"fake": FakeProvider()},
    )
    app = create_app(repo_root=tmp_path, manager=manager)
    client = TestClient(app)

    response = client.post(
        "/v1/video/generation-batches",
        json={
            "requests": [
                {
                    "provider": "fake",
                    "model": "fake-model",
                    "mode": "text_to_video",
                    "prompt": "first prompt",
                    "options": {"height": 64, "width": 64, "num_frames": 5},
                },
                {
                    "provider": "fake",
                    "model": "fake-model",
                    "mode": "text_to_video",
                    "prompt": "second prompt",
                    "options": {"height": 64, "width": 64, "num_frames": 5},
                },
            ],
            "metadata": {"purpose": "unit-test"},
        },
    )

    assert response.status_code == 202
    submitted = response.json()
    assert submitted["id"].startswith("batch_")
    assert submitted["metadata"] == {"purpose": "unit-test"}
    assert len(submitted["job_ids"]) == 2
    assert [job["id"] for job in submitted["jobs"]] == submitted["job_ids"]

    completed = wait_for_batch_status(client, submitted["id"], JobStatus.SUCCEEDED.value)
    assert completed["status"] == JobStatus.SUCCEEDED.value
    assert len(completed["jobs"]) == 2
    assert all(
        job["output"]["video_url"] == f"/v1/video/generations/{job['id']}/video"
        for job in completed["jobs"]
    )

    listed = client.get("/v1/video/generation-batches").json()
    assert [batch["id"] for batch in listed] == [submitted["id"]]


def test_video_backend_batch_rejects_invalid_request_without_partial_jobs(tmp_path: Path) -> None:
    manager = VideoJobManager(
        store=JobStore(tmp_path / "jobs"),
        providers={"fake": FakeProvider()},
    )
    app = create_app(repo_root=tmp_path, manager=manager)
    client = TestClient(app)

    response = client.post(
        "/v1/video/generation-batches",
        json={
            "requests": [
                {
                    "provider": "fake",
                    "model": "fake-model",
                    "mode": "text_to_video",
                    "prompt": "valid prompt",
                    "options": {"height": 64, "width": 64, "num_frames": 5},
                },
                {
                    "provider": "missing",
                    "model": "fake-model",
                    "mode": "text_to_video",
                    "prompt": "invalid prompt",
                    "options": {"height": 64, "width": 64, "num_frames": 5},
                },
            ],
        },
    )

    assert response.status_code == 400
    assert client.get("/v1/video/generations").json() == []
    assert client.get("/v1/video/generation-batches").json() == []


def test_disabled_remote_provider_is_visible_but_not_accepted(tmp_path: Path) -> None:
    app = create_app(repo_root=tmp_path)
    client = TestClient(app)

    providers = client.get("/v1/video/providers").json()["providers"]
    assert any(provider["id"] == "fal" and not provider["enabled"] for provider in providers)

    response = client.post(
        "/v1/video/generations",
        json={
            "provider": "fal",
            "model": "bytedance/seedance-2.0/image-to-video",
            "mode": "image_to_video",
            "prompt": "animate this",
            "image_url": "https://example.com/input.png",
        },
    )
    assert response.status_code == 501
