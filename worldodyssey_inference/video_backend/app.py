from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse

from worldodyssey_inference.video_backend.manager import UnknownProviderError, VideoJobManager
from worldodyssey_inference.video_backend.models import (
    JobStatus,
    ProviderListResponse,
    VideoGenerationBatchRecord,
    VideoGenerationBatchRequest,
    VideoGenerationRequest,
    VideoJobRecord,
)
from worldodyssey_inference.video_backend.providers import ProviderError, ProviderUnavailableError


def create_app(
    *,
    repo_root: Path | None = None,
    manager: VideoJobManager | None = None,
    max_workers: int = 1,
) -> FastAPI:
    resolved_root = repo_root or Path(__file__).resolve().parents[2]
    video_manager = manager or VideoJobManager.default(resolved_root, max_workers=max_workers)

    app = FastAPI(
        title="WorldOdyssey Video Backend",
        version="0.1.0",
        description="Provider-neutral video generation API for local SGLang and future remote providers.",
    )
    app.state.video_manager = video_manager

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/video/providers", response_model=ProviderListResponse)
    def list_providers() -> ProviderListResponse:
        return ProviderListResponse(providers=video_manager.capabilities())

    @app.post("/v1/video/generations", response_model=VideoJobRecord, status_code=202)
    def submit_generation(request: VideoGenerationRequest) -> VideoJobRecord:
        try:
            return video_manager.submit(request)
        except UnknownProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ProviderUnavailableError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/video/generation-batches", response_model=VideoGenerationBatchRecord, status_code=202)
    def submit_generation_batch(request: VideoGenerationBatchRequest) -> VideoGenerationBatchRecord:
        try:
            return video_manager.submit_batch(request)
        except UnknownProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ProviderUnavailableError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/video/generation-batches", response_model=list[VideoGenerationBatchRecord])
    def list_generation_batches(limit: int = Query(default=100, ge=1, le=500)) -> list[VideoGenerationBatchRecord]:
        return video_manager.list_batches(limit=limit)

    @app.get("/v1/video/generation-batches/{batch_id}", response_model=VideoGenerationBatchRecord)
    def get_generation_batch(batch_id: str) -> VideoGenerationBatchRecord:
        try:
            return video_manager.get_batch(batch_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown batch {batch_id!r}.") from exc

    @app.get("/v1/video/generations", response_model=list[VideoJobRecord])
    def list_generations(limit: int = Query(default=100, ge=1, le=500)) -> list[VideoJobRecord]:
        return video_manager.list(limit=limit)

    @app.get("/v1/video/generations/{job_id}", response_model=VideoJobRecord)
    def get_generation(job_id: str) -> VideoJobRecord:
        try:
            return video_manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id!r}.") from exc

    @app.get("/v1/video/generations/{job_id}/logs", response_class=PlainTextResponse)
    def get_generation_logs(job_id: str) -> PlainTextResponse:
        try:
            video_manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id!r}.") from exc
        return PlainTextResponse(video_manager.log_text(job_id))

    @app.get("/v1/video/generations/{job_id}/video")
    def get_generation_video(job_id: str) -> FileResponse:
        try:
            record = video_manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id!r}.") from exc
        if record.status != JobStatus.SUCCEEDED:
            raise HTTPException(status_code=409, detail=f"Job {job_id!r} is {record.status.value}.")
        video_path = video_manager.video_path(job_id)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video file for job {job_id!r} is missing.")
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=video_path.name,
        )

    return app
