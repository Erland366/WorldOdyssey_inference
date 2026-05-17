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

from worldodyssey_inference.video_backend.models import VideoGenerationBatchRequest
from worldodyssey_inference.video_backend.submission_config import (
    deep_merge,
    get_dotted_value,
    load_yaml_config,
    parse_dotted_overrides,
    set_dotted_value,
)
from worldodyssey_inference.video_backend.worldodyssey import (
    submit_generation_batch_request,
    wait_for_generation_batch,
)


@dataclass(frozen=True)
class BatchSubmissionPlan:
    backend_url: str
    request: VideoGenerationBatchRequest
    dry_run: bool
    wait: bool
    poll_interval_seconds: float
    wait_timeout_seconds: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a provider-neutral batch of video generation requests."
    )
    parser.add_argument("config", type=Path, help="YAML batch config.")
    parser.add_argument("--backend-url")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override a YAML value using dotted.path=value. List indexes are supported, for example requests.0.prompt=...",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print the /v1/video/generation-batches JSON payload without submitting.",
    )
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Poll the submitted batch until it reaches a terminal status.",
    )
    parser.add_argument("--poll-interval-seconds", type=float)
    parser.add_argument("--wait-timeout-seconds", type=int)
    return parser.parse_args(argv)


def build_batch_submission_plan(args: argparse.Namespace) -> BatchSubmissionPlan:
    config = load_yaml_config(args.config)
    _apply_cli_overrides(config, args)
    config = deep_merge(config, parse_dotted_overrides(args.overrides))

    run_config = _mapping_value(config, "run", {})
    request_payload = {
        key: config[key]
        for key in ("requests", "metadata")
        if key in config
    }
    request = VideoGenerationBatchRequest.model_validate(request_payload)
    backend_url = config.get("backend_url") or os.environ.get("VIDEO_BACKEND_URL", "http://127.0.0.1:8000")
    return BatchSubmissionPlan(
        backend_url=backend_url,
        request=request,
        dry_run=bool(run_config.get("dry_run", False)),
        wait=bool(run_config.get("wait", False)),
        poll_interval_seconds=float(run_config.get("poll_interval_seconds", 5.0)),
        wait_timeout_seconds=int(run_config.get("wait_timeout_seconds", 900)),
    )


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    direct_overrides = [
        ("backend_url", args.backend_url),
        ("run.dry_run", args.dry_run),
        ("run.wait", args.wait),
        ("run.poll_interval_seconds", args.poll_interval_seconds),
        ("run.wait_timeout_seconds", args.wait_timeout_seconds),
    ]
    for path, value in direct_overrides:
        if value is not None:
            set_dotted_value(config, path, value)


def _mapping_value(config: dict[str, Any], key: str, default: dict[str, Any]) -> dict[str, Any]:
    value = get_dotted_value(config, key, default)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected {key} to be a mapping.")
    return value


def main() -> int:
    args = parse_args()
    plan = build_batch_submission_plan(args)

    if plan.dry_run:
        print(json.dumps(plan.request.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0

    submitted = submit_generation_batch_request(plan.backend_url, plan.request)
    print(json.dumps(submitted, indent=2, sort_keys=True))

    if not plan.wait:
        return 0

    completed = wait_for_generation_batch(
        plan.backend_url,
        submitted["id"],
        poll_interval_seconds=plan.poll_interval_seconds,
        timeout_seconds=plan.wait_timeout_seconds,
    )
    print(json.dumps(completed, indent=2, sort_keys=True))
    if completed["status"] != "succeeded":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
