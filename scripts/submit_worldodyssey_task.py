#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldodyssey_inference.video_backend.models import (
    VideoGenerationBatchRequest,
    VideoGenerationRequest,
    VideoMode,
)
from worldodyssey_inference.video_backend.providers import (
    DEFAULT_SGLANG_I2V_MODEL,
)
from worldodyssey_inference.video_backend.submission_config import (
    deep_merge,
    get_dotted_value,
    load_yaml_config,
    parse_dotted_overrides,
    set_dotted_value,
)
from worldodyssey_inference.video_backend.worldodyssey import (
    DEFAULT_WORLDODYSSEY_TASK_DIR,
    build_worldodyssey_generation_request,
    discover_worldodyssey_tasks,
    download_generation_video,
    submit_generation_batch_request,
    submit_generation_request,
    wait_for_generation_batch,
    wait_for_generation,
)


@dataclass(frozen=True)
class SubmissionPlan:
    backend_url: str
    request: VideoGenerationRequest | None
    batch_request: VideoGenerationBatchRequest | None
    dry_run: bool
    wait: bool
    poll_interval_seconds: float
    wait_timeout_seconds: int
    download_path: Path | None
    download_dir: Path | None

    @property
    def is_batch(self) -> bool:
        return self.batch_request is not None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt WorldOdyssey task folders to the provider-neutral video backend API."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML submission config. CLI flags and --set overrides take precedence.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override a YAML value using dotted.path=value. Values are parsed as YAML scalars.",
    )
    parser.add_argument(
        "task",
        nargs="?",
        type=Path,
        default=None,
        help="WorldOdyssey task directory, task.json path, or parent directory containing task folders.",
    )
    parser.add_argument(
        "--backend-url",
        default=None,
    )
    parser.add_argument("--provider")
    parser.add_argument(
        "--model",
        default=None,
        help="Model id. Defaults to the local T2V model for text_to_video and the local I2V model for image_to_video.",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in VideoMode],
        default=None,
    )
    parser.add_argument(
        "--i2v",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=f"Shortcut for --mode image_to_video with the main task frame attached. Default I2V model: {DEFAULT_SGLANG_I2V_MODEL}.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Output height. Defaults are selected by mode.",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Output width. Defaults are selected by mode.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help="Number of output frames. Defaults are selected by mode.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        help="Optional provider field. Native local SGLang rejects this per request; configure steps on the server if supported.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional provider field. Native local SGLang rejects this per request.",
    )
    parser.add_argument(
        "--attention-backend",
        help="Optional provider field. Native local SGLang rejects this per request; set it on scripts/serve_sglang_diffusion.sh.",
    )
    parser.add_argument(
        "--vsa-sparsity",
        type=float,
        help="Optional provider field. Native local SGLang rejects this per request; set it on scripts/serve_sglang_diffusion.sh.",
    )
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument(
        "--include-main-image-base64",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Attach the main reference frame as image_base64 for future image-capable providers.",
    )
    parser.add_argument(
        "--prompt-prefix",
        help="Text to prepend before the WorldOdyssey task prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print the generated /v1/video/generations JSON payload without submitting.",
    )
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Poll the submitted job until it reaches a terminal status.",
    )
    parser.add_argument("--poll-interval-seconds", type=float)
    parser.add_argument("--wait-timeout-seconds", type=int)
    parser.add_argument(
        "--download-path",
        type=Path,
        help="Download the generated video after a waited job succeeds.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Download generated batch videos after a waited batch succeeds. Filenames use task ids.",
    )
    return parser.parse_args(argv)


def build_submission_plan(args: argparse.Namespace) -> SubmissionPlan:
    config = load_yaml_config(args.config)
    _apply_cli_overrides(config, args)
    config = deep_merge(config, parse_dotted_overrides(args.overrides))

    request_config = _mapping_value(config, "request", {})
    options_config = _mapping_value(request_config, "options", {})
    adapter_config = _mapping_value(config, "adapter", {})
    run_config = _mapping_value(config, "run", {})

    i2v = bool(adapter_config.get("i2v", False))
    mode = _resolve_mode(request_config.get("mode"), i2v=i2v)
    task_path = _optional_path(config.get("task")) or DEFAULT_WORLDODYSSEY_TASK_DIR
    task_selection = discover_worldodyssey_tasks(task_path)

    include_main_image_base64 = _resolve_include_main_image(
        adapter_config.get("include_main_image_base64"),
        i2v=i2v,
        mode=mode,
        request_config=request_config,
    )
    requests = [
        _build_request_for_task(
            loaded_task,
            request_config=request_config,
            options_config=options_config,
            adapter_config=adapter_config,
            mode=mode,
            include_main_image_base64=include_main_image_base64,
        )
        for loaded_task in task_selection.tasks
    ]

    backend_url = config.get("backend_url") or os.environ.get("VIDEO_BACKEND_URL", "http://127.0.0.1:8000")
    download_path = _optional_path(run_config.get("download_path"))
    download_dir = _optional_path(run_config.get("download_dir"))
    if task_selection.from_parent_directory:
        if download_path is not None:
            raise ValueError("Batch WorldOdyssey submissions use run.download_dir, not run.download_path.")
        batch_request = VideoGenerationBatchRequest.model_validate(
            {
                "requests": [request.model_dump(mode="json") for request in requests],
                "metadata": {
                    "adapter": "worldodyssey",
                    "task_root": str(task_selection.root_path),
                    "task_ids": [task.task_id for task in task_selection.tasks],
                    "skipped_entries": [str(path) for path in task_selection.skipped_entries],
                },
            }
        )
        return SubmissionPlan(
            backend_url=backend_url,
            request=None,
            batch_request=batch_request,
            dry_run=bool(run_config.get("dry_run", False)),
            wait=bool(run_config.get("wait", False)),
            poll_interval_seconds=float(run_config.get("poll_interval_seconds", 5.0)),
            wait_timeout_seconds=int(run_config.get("wait_timeout_seconds", 900)),
            download_path=None,
            download_dir=download_dir,
        )

    if download_dir is not None:
        raise ValueError("Single-task WorldOdyssey submissions use run.download_path, not run.download_dir.")

    return SubmissionPlan(
        backend_url=backend_url,
        request=requests[0],
        batch_request=None,
        dry_run=bool(run_config.get("dry_run", False)),
        wait=bool(run_config.get("wait", False)),
        poll_interval_seconds=float(run_config.get("poll_interval_seconds", 5.0)),
        wait_timeout_seconds=int(run_config.get("wait_timeout_seconds", 900)),
        download_path=download_path,
        download_dir=download_dir,
    )


def _build_request_for_task(
    loaded_task,
    *,
    request_config: dict[str, Any],
    options_config: dict[str, Any],
    adapter_config: dict[str, Any],
    mode: VideoMode,
    include_main_image_base64: bool,
) -> VideoGenerationRequest:
    base_request = build_worldodyssey_generation_request(
        loaded_task,
        provider=request_config.get("provider", "sglang"),
        model=request_config.get("model"),
        mode=mode,
        height=options_config.get("height"),
        width=options_config.get("width"),
        num_frames=options_config.get("num_frames"),
        num_inference_steps=options_config.get("num_inference_steps"),
        seed=options_config.get("seed"),
        attention_backend=options_config.get("attention_backend"),
        vsa_sparsity=options_config.get("vsa_sparsity"),
        timeout_seconds=options_config.get("timeout_seconds", 300),
        include_main_image_base64=include_main_image_base64,
        prompt_prefix=adapter_config.get("prompt_prefix"),
    )
    request_payload = deep_merge(base_request.model_dump(mode="json"), request_config)
    return VideoGenerationRequest.model_validate(request_payload)


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    direct_overrides = [
        ("task", _string_path(args.task)),
        ("backend_url", args.backend_url),
        ("request.provider", args.provider),
        ("request.model", args.model),
        ("request.mode", args.mode),
        ("request.options.height", args.height),
        ("request.options.width", args.width),
        ("request.options.num_frames", args.num_frames),
        ("request.options.num_inference_steps", args.num_inference_steps),
        ("request.options.seed", args.seed),
        ("request.options.attention_backend", args.attention_backend),
        ("request.options.vsa_sparsity", args.vsa_sparsity),
        ("request.options.timeout_seconds", args.timeout_seconds),
        ("adapter.i2v", args.i2v),
        ("adapter.include_main_image_base64", args.include_main_image_base64),
        ("adapter.prompt_prefix", args.prompt_prefix),
        ("run.dry_run", args.dry_run),
        ("run.wait", args.wait),
        ("run.poll_interval_seconds", args.poll_interval_seconds),
        ("run.wait_timeout_seconds", args.wait_timeout_seconds),
        ("run.download_path", _string_path(args.download_path)),
        ("run.download_dir", _string_path(args.download_dir)),
    ]
    for path, value in direct_overrides:
        if value is not None:
            set_dotted_value(config, path, value)
    if args.i2v is True and args.mode is None:
        set_dotted_value(config, "request.mode", VideoMode.IMAGE_TO_VIDEO.value)


def _mapping_value(config: dict[str, Any], key: str, default: dict[str, Any]) -> dict[str, Any]:
    value = get_dotted_value(config, key, default)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected {key} to be a mapping.")
    return value


def _resolve_mode(value: Any, *, i2v: bool) -> VideoMode:
    if value is None:
        return VideoMode.IMAGE_TO_VIDEO if i2v else VideoMode.TEXT_TO_VIDEO
    return VideoMode(value)


def _resolve_include_main_image(
    value: Any,
    *,
    i2v: bool,
    mode: VideoMode,
    request_config: dict[str, Any],
) -> bool:
    if value is not None:
        return bool(value)
    has_image_input = any(
        request_config.get(field)
        for field in ("image_path", "image_url", "image_base64", "end_image_url", "reference_image_urls")
    )
    return i2v or (mode == VideoMode.IMAGE_TO_VIDEO and not has_image_input)


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _string_path(value: Path | None) -> str | None:
    if value is None:
        return None
    return str(value)


def main() -> int:
    args = parse_args()
    plan = build_submission_plan(args)

    if plan.dry_run:
        request = _planned_payload(plan)
        print(json.dumps(request.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0

    if plan.batch_request is not None:
        submitted = submit_generation_batch_request(plan.backend_url, plan.batch_request)
    elif plan.request is not None:
        submitted = submit_generation_request(plan.backend_url, plan.request)
    else:
        raise AssertionError("Submission plan has neither a request nor a batch request.")
    print(json.dumps(submitted, indent=2, sort_keys=True))

    if not plan.wait:
        return 0

    if plan.batch_request is not None:
        completed = wait_for_generation_batch(
            plan.backend_url,
            submitted["id"],
            poll_interval_seconds=plan.poll_interval_seconds,
            timeout_seconds=plan.wait_timeout_seconds,
        )
    else:
        completed = wait_for_generation(
            plan.backend_url,
            submitted["id"],
            poll_interval_seconds=plan.poll_interval_seconds,
            timeout_seconds=plan.wait_timeout_seconds,
        )
    print(json.dumps(completed, indent=2, sort_keys=True))
    if completed["status"] == "succeeded" and plan.download_path:
        download_generation_video(plan.backend_url, submitted["id"], plan.download_path)
        print(f"Downloaded video to {plan.download_path}")
    if completed["status"] == "succeeded" and plan.batch_request is not None and plan.download_dir:
        downloaded_paths = _download_batch_videos(plan.backend_url, completed, plan.download_dir)
        for downloaded_path in downloaded_paths:
            print(f"Downloaded video to {downloaded_path}")
    return 0


def _planned_payload(plan: SubmissionPlan) -> VideoGenerationRequest | VideoGenerationBatchRequest:
    if plan.batch_request is not None:
        return plan.batch_request
    if plan.request is not None:
        return plan.request
    raise AssertionError("Submission plan has neither a request nor a batch request.")


def _download_batch_videos(backend_url: str, completed_batch: dict[str, Any], download_dir: Path) -> list[Path]:
    downloaded_paths: list[Path] = []
    used_names: set[str] = set()
    for job in completed_batch["jobs"]:
        if job["status"] != "succeeded":
            continue
        metadata = job["request"].get("metadata", {})
        task_id = metadata.get("task_id") or job["id"]
        filename = f"{task_id}.mp4"
        if filename in used_names:
            filename = f"{task_id}-{job['id']}.mp4"
        used_names.add(filename)
        output_path = download_dir / filename
        download_generation_video(backend_url, job["id"], output_path)
        downloaded_paths.append(output_path)
    return downloaded_paths


if __name__ == "__main__":
    raise SystemExit(main())
