from __future__ import annotations

import json
import re
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from worldodyssey_inference.video_backend.models import VideoJobRecord


JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class JobPaths:
    job_id: str
    job_dir: Path
    output_path: Path
    log_path: Path
    record_path: Path


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.records_dir = root / "jobs"
        self.videos_dir = root / "videos"
        self.logs_dir = root / "logs"
        self._lock = threading.Lock()
        for directory in (self.records_dir, self.videos_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def new_job_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"vid_{timestamp}_{secrets.token_hex(4)}"

    def paths_for(self, job_id: str) -> JobPaths:
        self._validate_job_id(job_id)
        job_dir = self.videos_dir / job_id
        return JobPaths(
            job_id=job_id,
            job_dir=job_dir,
            output_path=job_dir / "output.mp4",
            log_path=self.logs_dir / f"{job_id}.log",
            record_path=self.records_dir / f"{job_id}.json",
        )

    def write(self, record: VideoJobRecord) -> None:
        paths = self.paths_for(record.id)
        paths.record_path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.model_dump(mode="json")
        tmp_path = paths.record_path.with_suffix(".json.tmp")
        with self._lock:
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp_path.replace(paths.record_path)

    def read(self, job_id: str) -> VideoJobRecord:
        paths = self.paths_for(job_id)
        if not paths.record_path.exists():
            raise KeyError(job_id)
        return VideoJobRecord.model_validate_json(paths.record_path.read_text(encoding="utf-8"))

    def list_records(self, *, limit: int = 100) -> list[VideoJobRecord]:
        records: list[VideoJobRecord] = []
        for path in sorted(self.records_dir.glob("*.json"), reverse=True):
            records.append(VideoJobRecord.model_validate_json(path.read_text(encoding="utf-8")))
            if len(records) >= limit:
                break
        return records

    @staticmethod
    def _validate_job_id(job_id: str) -> None:
        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise KeyError(job_id)
