from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from worldodyssey_inference.video_backend.models import (
    ErrorInfo,
    JobStatus,
    ProviderCapability,
    VideoGenerationRequest,
    VideoJobRecord,
    VideoOutput,
    utc_now_iso,
)
from worldodyssey_inference.video_backend.providers import (
    ProviderError,
    VideoProvider,
    default_providers,
)
from worldodyssey_inference.video_backend.storage import JobStore


class UnknownProviderError(ValueError):
    pass


class VideoJobManager:
    def __init__(
        self,
        *,
        store: JobStore,
        providers: dict[str, VideoProvider],
        max_workers: int = 1,
    ) -> None:
        self.store = store
        self.providers = providers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="video-backend")

    @classmethod
    def default(cls, repo_root: Path, *, max_workers: int = 1) -> VideoJobManager:
        return cls(
            store=JobStore(repo_root / "artifacts" / "video-backend"),
            providers=default_providers(repo_root),
            max_workers=max_workers,
        )

    def capabilities(self) -> list[ProviderCapability]:
        return [provider.capability for provider in self.providers.values()]

    def submit(self, request: VideoGenerationRequest) -> VideoJobRecord:
        provider = self.providers.get(request.provider)
        if provider is None:
            raise UnknownProviderError(f"Unknown provider {request.provider!r}.")
        provider.validate_request(request)
        now = utc_now_iso()
        job_id = self.store.new_job_id()
        paths = self.store.paths_for(job_id)
        record = VideoJobRecord(
            id=job_id,
            status=JobStatus.QUEUED,
            provider=request.provider,
            model=request.model,
            mode=request.mode,
            created_at=now,
            updated_at=now,
            request=request,
            log_path=str(paths.log_path),
        )
        self.store.write(record)
        future = self.executor.submit(self._run_job, job_id)
        future.add_done_callback(self._raise_unexpected_worker_error)
        return record

    def get(self, job_id: str) -> VideoJobRecord:
        return self.store.read(job_id)

    def list(self, *, limit: int = 100) -> list[VideoJobRecord]:
        return self.store.list_records(limit=limit)

    def video_path(self, job_id: str) -> Path:
        return self.store.paths_for(job_id).output_path

    def log_text(self, job_id: str) -> str:
        log_path = self.store.paths_for(job_id).log_path
        if not log_path.exists():
            return ""
        return log_path.read_text(encoding="utf-8", errors="replace")

    def _run_job(self, job_id: str) -> None:
        record = self.store.read(job_id)
        provider = self.providers[record.provider]
        paths = self.store.paths_for(job_id)
        record.status = JobStatus.RUNNING
        record.updated_at = utc_now_iso()
        self.store.write(record)

        try:
            result = provider.run(record, paths)
            stat = result.output_path.stat()
            record.status = JobStatus.SUCCEEDED
            record.output = VideoOutput(
                video_url=f"/v1/video/generations/{record.id}/video",
                local_path=str(result.output_path),
                file_size=stat.st_size,
            )
            record.metrics.update(result.metrics)
            record.error = None
        except ProviderError as exc:
            record.status = JobStatus.FAILED
            record.error = ErrorInfo(
                code=exc.code,
                message=str(exc),
                retryable=exc.retryable,
            )
        finally:
            record.updated_at = utc_now_iso()
            self.store.write(record)

    @staticmethod
    def _raise_unexpected_worker_error(future: Future[None]) -> None:
        future.result()
