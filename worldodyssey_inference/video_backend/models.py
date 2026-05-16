from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VideoMode(str, Enum):
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    REFERENCE_TO_VIDEO = "reference_to_video"
    EDIT_VIDEO = "edit_video"
    EXTEND_VIDEO = "extend_video"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoGenerationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    duration: int | None = Field(default=None, ge=1)
    num_frames: int | None = Field(default=None, ge=1)
    resolution: str | None = None
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    aspect_ratio: str | None = None
    fps: int | None = Field(default=None, ge=1)
    seed: int | None = None
    guidance_scale: float | None = None
    num_inference_steps: int | None = Field(default=None, ge=1)
    num_gpus: int = Field(default=1, ge=1)
    generate_audio: bool | None = None
    attention_backend: str | None = None
    vsa_sparsity: float | None = Field(default=None, ge=0.0, le=1.0)
    timeout_seconds: int = Field(default=300, ge=1)
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    provider_options: dict[str, Any] = Field(default_factory=dict)


class VideoGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "sglang"
    model: str = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
    mode: VideoMode = VideoMode.TEXT_TO_VIDEO
    prompt: str
    negative_prompt: str | None = None
    image_url: str | None = None
    image_base64: str | None = None
    end_image_url: str | None = None
    reference_image_urls: list[str] = Field(default_factory=list)
    video_url: str | None = None
    options: VideoGenerationOptions = Field(default_factory=VideoGenerationOptions)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoOutput(BaseModel):
    video_url: str | None = None
    local_path: str | None = None
    content_type: str = "video/mp4"
    file_size: int | None = None


class ErrorInfo(BaseModel):
    code: str
    message: str
    provider_code: str | None = None
    retryable: bool = False


class VideoJobRecord(BaseModel):
    id: str
    status: JobStatus
    provider: str
    model: str
    mode: VideoMode
    created_at: str
    updated_at: str
    request: VideoGenerationRequest
    output: VideoOutput | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: ErrorInfo | None = None
    log_path: str | None = None


class ProviderCapability(BaseModel):
    id: str
    label: str
    enabled: bool
    local: bool
    models: list[str]
    modes: list[VideoMode]
    supports_audio: bool = False
    supports_seed: bool = False
    supports_custom_resolution: bool = False
    supports_reference_images: bool = False
    supports_end_image: bool = False
    resolutions: list[str] = Field(default_factory=list)
    setup: dict[str, Any] = Field(default_factory=dict)


class ProviderListResponse(BaseModel):
    providers: list[ProviderCapability]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
