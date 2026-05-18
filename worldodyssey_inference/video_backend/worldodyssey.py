from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from worldodyssey_inference.video_backend.models import (
    VideoGenerationBatchRequest,
    VideoGenerationRequest,
    VideoMode,
)
from worldodyssey_inference.video_backend.providers import (
    DEFAULT_SGLANG_HEIGHT,
    DEFAULT_SGLANG_I2V_HEIGHT,
    DEFAULT_SGLANG_I2V_MODEL,
    DEFAULT_SGLANG_I2V_NUM_FRAMES,
    DEFAULT_SGLANG_I2V_WIDTH,
    DEFAULT_SGLANG_MODEL,
    DEFAULT_SGLANG_NUM_FRAMES,
    DEFAULT_SGLANG_WIDTH,
)


DEFAULT_WORLDODYSSEY_TASK_DIR = (
    Path(__file__).resolve().parents[2]
    / "submodule"
    / "worldodyssey"
    / "inputs"
    / "move_bookmark"
)


class WorldOdysseyFrames(BaseModel):
    model_config = ConfigDict(extra="forbid")

    main: str
    extras: list[str] = Field(default_factory=list)


class WorldOdysseyEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: str
    objects: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class WorldOdysseyTopologyGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    states: dict[str, str]
    edges: dict[str, WorldOdysseyEdge]
    valid_topo_sorts: list[list[str]]
    start_state: str
    goal_state: str


class WorldOdysseyTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    frames: WorldOdysseyFrames
    topology_graph: WorldOdysseyTopologyGraph


@dataclass(frozen=True)
class LoadedWorldOdysseyTask:
    task_id: str
    task_dir: Path
    task_json_path: Path
    task: WorldOdysseyTask
    frame_paths: tuple[Path, ...]
    source_video_paths: dict[str, Path]


@dataclass(frozen=True)
class WorldOdysseyTaskSelection:
    root_path: Path
    tasks: tuple[LoadedWorldOdysseyTask, ...]
    skipped_entries: tuple[Path, ...]
    from_parent_directory: bool


def resolve_worldodyssey_task_dir(path: Path | None = None) -> Path:
    selected_path = path or DEFAULT_WORLDODYSSEY_TASK_DIR
    resolved = selected_path.expanduser().resolve()
    if resolved.is_file():
        if resolved.name != "task.json":
            raise ValueError(f"Expected a WorldOdyssey task directory or task.json, got file: {resolved}")
        return resolved.parent
    return resolved


def discover_worldodyssey_tasks(path: Path | None = None) -> WorldOdysseyTaskSelection:
    selected_path = path or DEFAULT_WORLDODYSSEY_TASK_DIR
    resolved = selected_path.expanduser().resolve()
    if resolved.is_file() or (resolved / "task.json").exists():
        return WorldOdysseyTaskSelection(
            root_path=resolve_worldodyssey_task_dir(resolved),
            tasks=(load_worldodyssey_task(resolved),),
            skipped_entries=(),
            from_parent_directory=False,
        )

    if not resolved.exists():
        raise FileNotFoundError(f"WorldOdyssey input path not found: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Expected a WorldOdyssey task path or parent directory, got: {resolved}")

    task_dirs: list[Path] = []
    skipped_entries: list[Path] = []
    for child in sorted(resolved.iterdir(), key=lambda item: item.name):
        if child.is_dir() and (child / "task.json").exists():
            task_dirs.append(child)
        else:
            skipped_entries.append(child)

    if not task_dirs:
        raise FileNotFoundError(
            f"No WorldOdyssey task.json files found in {resolved}. "
            "Pass a task directory, a task.json file, or a parent directory containing task folders."
        )

    return WorldOdysseyTaskSelection(
        root_path=resolved,
        tasks=tuple(load_worldodyssey_task(task_dir) for task_dir in task_dirs),
        skipped_entries=tuple(skipped_entries),
        from_parent_directory=True,
    )


def load_worldodyssey_task(path: Path | None = None) -> LoadedWorldOdysseyTask:
    task_dir = resolve_worldodyssey_task_dir(path)
    task_json_path = task_dir / "task.json"
    if not task_json_path.exists():
        raise FileNotFoundError(f"WorldOdyssey task.json not found: {task_json_path}")

    task_payload = json.loads(task_json_path.read_text(encoding="utf-8"))
    task = WorldOdysseyTask.model_validate(task_payload)
    frame_paths = _collect_frame_paths(task_dir, task)
    source_video_paths = {
        stem: video_path
        for stem in ("ai", "gt")
        if (video_path := task_dir / f"{stem}.mp4").exists()
    }
    return LoadedWorldOdysseyTask(
        task_id=task_dir.name,
        task_dir=task_dir,
        task_json_path=task_json_path,
        task=task,
        frame_paths=tuple(frame_paths),
        source_video_paths=source_video_paths,
    )


def build_worldodyssey_prompt(
    loaded_task: LoadedWorldOdysseyTask,
    *,
    prompt_prefix: str | None = None,
) -> str:
    task_prompt = loaded_task.task.task.strip()
    if not prompt_prefix:
        return task_prompt

    prefix = prompt_prefix.strip()
    if not prefix:
        return task_prompt
    return f"{prefix}\n{task_prompt}"


def build_worldodyssey_generation_request(
    loaded_task: LoadedWorldOdysseyTask,
    *,
    provider: str = "sglang",
    model: str | None = None,
    mode: VideoMode = VideoMode.TEXT_TO_VIDEO,
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
    num_inference_steps: int | None = None,
    seed: int | None = None,
    attention_backend: str | None = None,
    vsa_sparsity: float | None = None,
    timeout_seconds: int = 300,
    include_main_image_base64: bool = False,
    prompt_prefix: str | None = None,
) -> VideoGenerationRequest:
    selected_model = model or (
        DEFAULT_SGLANG_I2V_MODEL if mode == VideoMode.IMAGE_TO_VIDEO else DEFAULT_SGLANG_MODEL
    )
    if mode == VideoMode.IMAGE_TO_VIDEO:
        height = height or DEFAULT_SGLANG_I2V_HEIGHT
        width = width or DEFAULT_SGLANG_I2V_WIDTH
        num_frames = num_frames or DEFAULT_SGLANG_I2V_NUM_FRAMES
    else:
        height = height or DEFAULT_SGLANG_HEIGHT
        width = width or DEFAULT_SGLANG_WIDTH
        num_frames = num_frames or DEFAULT_SGLANG_NUM_FRAMES

    metadata: dict[str, Any] = {
        "adapter": "worldodyssey",
        "task_id": loaded_task.task_id,
        "task_dir": str(loaded_task.task_dir),
        "task_json_path": str(loaded_task.task_json_path),
        "frame_paths": [str(path) for path in loaded_task.frame_paths],
        "source_video_paths": {
            key: str(path) for key, path in loaded_task.source_video_paths.items()
        },
        "topology_graph": loaded_task.task.topology_graph.model_dump(mode="json"),
    }
    payload: dict[str, Any] = {
        "provider": provider,
        "mode": mode,
        "prompt": build_worldodyssey_prompt(loaded_task, prompt_prefix=prompt_prefix),
        "options": {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "attention_backend": attention_backend,
            "vsa_sparsity": vsa_sparsity,
            "timeout_seconds": timeout_seconds,
        },
        "metadata": metadata,
    }
    payload["model"] = selected_model

    if include_main_image_base64:
        payload["image_base64"] = encode_file_base64(loaded_task.frame_paths[0])

    return VideoGenerationRequest.model_validate(payload)


def encode_file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def submit_generation_request(backend_url: str, request: VideoGenerationRequest) -> dict[str, Any]:
    url = backend_url.rstrip("/") + "/v1/video/generations"
    return _request_json("POST", url, request.model_dump(mode="json"))


def submit_generation_batch_request(backend_url: str, request: VideoGenerationBatchRequest) -> dict[str, Any]:
    url = backend_url.rstrip("/") + "/v1/video/generation-batches"
    return _request_json("POST", url, request.model_dump(mode="json"))


def get_generation(backend_url: str, job_id: str) -> dict[str, Any]:
    url = backend_url.rstrip("/") + f"/v1/video/generations/{job_id}"
    return _request_json("GET", url)


def get_generation_batch(backend_url: str, batch_id: str) -> dict[str, Any]:
    url = backend_url.rstrip("/") + f"/v1/video/generation-batches/{batch_id}"
    return _request_json("GET", url)


def wait_for_generation(
    backend_url: str,
    job_id: str,
    *,
    poll_interval_seconds: float = 5.0,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        record = get_generation(backend_url, job_id)
        if record["status"] in {"succeeded", "failed", "cancelled"}:
            return record
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"Timed out waiting for video generation job {job_id!r}.")


def wait_for_generation_batch(
    backend_url: str,
    batch_id: str,
    *,
    poll_interval_seconds: float = 5.0,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        record = get_generation_batch(backend_url, batch_id)
        if record["status"] in {"succeeded", "failed", "cancelled"}:
            return record
        time.sleep(poll_interval_seconds)
    raise TimeoutError(f"Timed out waiting for video generation batch {batch_id!r}.")


def download_generation_video(backend_url: str, job_id: str, output_path: Path) -> None:
    url = backend_url.rstrip("/") + f"/v1/video/generations/{job_id}/video"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=60) as response:
        output_path.write_bytes(response.read())


def _collect_frame_paths(task_dir: Path, task: WorldOdysseyTask) -> list[Path]:
    relative_paths = [task.frames.main, *task.frames.extras]
    frame_paths = [task_dir / relative_path for relative_path in relative_paths]
    missing = [path for path in frame_paths if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"WorldOdyssey frame path(s) not found: {joined}")
    return frame_paths


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {error_body}") from exc
    return json.loads(response_body)
