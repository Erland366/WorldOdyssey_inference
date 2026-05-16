from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from worldodyssey_inference.video_backend.app import create_app
from worldodyssey_inference.video_backend.manager import VideoJobManager
from worldodyssey_inference.video_backend.models import (
    JobStatus,
    ProviderCapability,
    VideoGenerationRequest,
    VideoMode,
)
from worldodyssey_inference.video_backend.providers import (
    DEFAULT_SGLANG_MODEL,
    LocalSGLangProvider,
    ProviderRunResult,
    UnsupportedRequestError,
)
from worldodyssey_inference.video_backend.storage import JobPaths, JobStore


def make_sglang_runtime(root: Path) -> Path:
    venv = root / ".venv_sglangcuda12"
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
            "num_inference_steps": 3,
            "seed": 123,
            "attention_backend": "video_sparse_attn",
            "vsa_sparsity": 0.5,
        },
    }
    payload.update(overrides)
    return VideoGenerationRequest.model_validate(payload)


def test_sglang_command_maps_unified_request_to_cli(tmp_path: Path) -> None:
    provider = LocalSGLangProvider(repo_root=tmp_path, venv_path=make_sglang_runtime(tmp_path))
    output_path = tmp_path / "artifacts" / "video-backend" / "videos" / "job" / "output.mp4"

    command = provider.build_command(make_request(), output_path)

    assert command[:2] == ["sglang", "generate"]
    assert command[command.index("--model-path") + 1] == DEFAULT_SGLANG_MODEL
    assert "--attention-backend=video_sparse_attn" in command
    assert "--VSA-sparsity=0.5" in command
    assert "--height=448" in command
    assert "--width=832" in command
    assert "--num-frames=61" in command
    assert "--num-inference-steps=3" in command
    assert command[command.index("--output-path") + 1] == str(output_path.parent)
    assert command[command.index("--output-file-name") + 1] == "output.mp4"


def test_sglang_runtime_env_avoids_miniconda_compiler_paths(tmp_path: Path, monkeypatch) -> None:
    provider = LocalSGLangProvider(repo_root=tmp_path, venv_path=make_sglang_runtime(tmp_path))
    monkeypatch.setenv("PATH", "/home/coder/miniconda3/bin:/usr/bin:/bin")

    env = provider.runtime_env()

    assert env["PATH"] == f"{provider.venv_path / 'bin'}:/usr/local/bin:/usr/bin:/bin"
    assert env["CC"] == "/usr/bin/gcc"
    assert env["CXX"] == "/usr/bin/g++"
    assert env["CUDA_HOME"] == str(provider.cuda_home)


def test_sglang_provider_fails_fast_for_unsupported_modes(tmp_path: Path) -> None:
    provider = LocalSGLangProvider(repo_root=tmp_path, venv_path=make_sglang_runtime(tmp_path))
    request = make_request(mode="image_to_video", image_url="https://example.com/input.png")

    with pytest.raises(UnsupportedRequestError, match="text_to_video only"):
        provider.validate_request(request)


def test_sglang_provider_requires_explicit_vsa_options(tmp_path: Path) -> None:
    provider = LocalSGLangProvider(repo_root=tmp_path, venv_path=make_sglang_runtime(tmp_path))
    request = make_request(options={"height": 448, "width": 832, "num_frames": 61})

    with pytest.raises(UnsupportedRequestError, match="attention_backend"):
        provider.validate_request(request)


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
