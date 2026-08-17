#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
DEFAULT_MODEL = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
DEFAULT_PROMPT = (
    "A small matte red cube rotates slowly on a clean light gray studio floor, "
    "centered composition, soft cinematic lighting, realistic shadows, static camera"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a quality-profile FastWan request to the WorldOdyssey unified video API."
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("VIDEO_BACKEND_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--model", default=os.environ.get("FASTWAN_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get(
                "FASTWAN_OUTPUT",
                "artifacts/backend-videos/fastwan-sglang-quality.mp4",
            )
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=61)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def open_json(request: Request, timeout: float) -> dict[str, object]:
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Backend returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise SystemExit(
            "Unified video backend is not reachable. Start the native FastWan server and "
            f"scripts/serve_video_backend.py first: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit("Backend returned a non-object JSON response.")
    return payload


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "provider": "sglang",
        "model": args.model,
        "mode": "text_to_video",
        "prompt": args.prompt,
        "options": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "num_inference_steps": args.steps,
            "seed": args.seed,
            "timeout_seconds": args.timeout_seconds,
        },
        "metadata": {"helper": "scripts/run_fastwan_quality.sh"},
    }


def main() -> int:
    args = parse_args()
    base_url = args.backend_url.rstrip("/")
    payload = build_payload(args)
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    request = Request(
        f"{base_url}/v1/video/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    job = open_json(request, timeout=30)
    job_id = job.get("id")
    if not isinstance(job_id, str):
        raise SystemExit(f"Backend response is missing a job id: {job}")
    print(f"Submitted FastWan job {job_id}")

    deadline = time.monotonic() + args.timeout_seconds
    while True:
        job = open_json(Request(f"{base_url}/v1/video/generations/{job_id}"), timeout=30)
        status = job.get("status")
        print(f"{job_id}: {status}", flush=True)
        if status in TERMINAL_STATUSES:
            break
        if time.monotonic() >= deadline:
            raise SystemExit(f"Timed out waiting for FastWan job {job_id}")
        time.sleep(args.poll_interval_seconds)

    if status != "succeeded":
        raise SystemExit(json.dumps(job.get("error") or job, indent=2, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(f"{base_url}/v1/video/generations/{job_id}/video", timeout=60) as response:
            args.output.write_bytes(response.read())
    except (HTTPError, URLError) as exc:
        raise SystemExit(f"Could not download completed video: {exc}") from exc

    print(f"Downloaded {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
