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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit Cosmos 3 text-to-video or image-to-video to the WorldOdyssey unified video API."
    )
    parser.add_argument("--backend-url", default=os.environ.get("VIDEO_BACKEND_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default=os.environ.get("COSMOS3_MODEL", "nvidia/Cosmos3-Nano"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("COSMOS3_OUTPUT", "artifacts/backend-videos/cosmos3-nano.mp4")),
    )
    parser.add_argument("--prompt", default="A mobile robot navigates a warehouse aisle and stops at a shelf.")
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--image-path",
        type=Path,
        help="Local starting image. Supplying it switches the request to image-to-video.",
    )
    image_group.add_argument(
        "--image-url",
        help="HTTP(S) starting-image URL. Supplying it switches the request to image-to-video.",
    )
    parser.add_argument("--negative-prompt", default="blurry, distorted, low quality")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--flow-shift", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--guardrails", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
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
            "Unified video backend is not reachable. Start scripts/serve_video_backend.py first: "
            f"{exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit("Backend returned a non-object JSON response.")
    return payload


def main() -> int:
    args = parse_args()
    base_url = args.backend_url.rstrip("/")
    mode = "image_to_video" if args.image_path is not None or args.image_url else "text_to_video"
    payload = {
        "provider": "sglang",
        "model": args.model,
        "mode": mode,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "options": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "generate_audio": args.generate_audio,
            "timeout_seconds": args.timeout_seconds,
            "provider_options": {
                "request_fields": {"flow_shift": args.flow_shift},
                "extra_params": {
                    "guardrails": args.guardrails,
                    "use_resolution_template": False,
                    "use_duration_template": False,
                },
            },
        },
        "metadata": {"helper": "scripts/generate_cosmos3.sh"},
    }
    if args.image_path is not None:
        image_path = args.image_path.expanduser().resolve()
        if not image_path.is_file():
            raise SystemExit(f"Starting image does not exist or is not a file: {image_path}")
        payload["image_path"] = str(image_path)
    elif args.image_url:
        payload["image_url"] = args.image_url
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
    print(f"Submitted Cosmos 3 job {job_id}")

    deadline = time.monotonic() + args.timeout_seconds
    while True:
        job = open_json(Request(f"{base_url}/v1/video/generations/{job_id}"), timeout=30)
        status = job.get("status")
        print(f"{job_id}: {status}", flush=True)
        if status in TERMINAL_STATUSES:
            break
        if time.monotonic() >= deadline:
            raise SystemExit(f"Timed out waiting for Cosmos 3 job {job_id}")
        time.sleep(args.poll_interval_seconds)

    if status != "succeeded":
        raise SystemExit(json.dumps(job.get("error") or job, indent=2, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(f"{base_url}/v1/video/generations/{job_id}/video", timeout=60) as response:
            args.output.write_bytes(response.read())
    except (HTTPError, URLError) as exc:
        raise SystemExit(f"Could not download completed video: {exc}") from exc
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
