from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the WorldOdyssey provider-neutral video backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn worker count. Keep this at 1 for local GPUs.")
    parser.add_argument("--job-workers", type=int, default=1, help="Concurrent generation jobs. Keep this at 1 per GPU host.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("uvicorn is not installed. Run scripts/setup_video_backend.sh first.") from exc

    from worldodyssey_inference.video_backend.app import create_app

    app = create_app(repo_root=args.repo_root, max_workers=args.job_workers)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
