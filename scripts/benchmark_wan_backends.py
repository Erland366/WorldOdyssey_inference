from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worldodyssey_inference.wan_benchmark import (
    BENCHMARK_CASES,
    DEFAULT_FASTVIDEO_ATTENTION_BACKEND,
    DEFAULT_PROMPT,
    default_output_dir,
    default_video_dir,
    expand_matrix,
    record_path_for_request,
    run_preflight,
    run_preflight_subprocess,
    run_subprocess_request,
    run_worker_request_file,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Wan Diffusers and FastVideo inference backends.")
    parser.add_argument("--stage", choices=["fit", "benchmark", "all"], default="all")
    parser.add_argument("--case", choices=sorted(BENCHMARK_CASES), nargs="+", default=None)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--fastvideo-attention-backend", default=DEFAULT_FASTVIDEO_ATTENTION_BACKEND)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-fit-video", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--worker-config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-preflight", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_parent(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or default_output_dir(repo_root)
    video_dir = args.video_dir or default_video_dir(repo_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    if not args.skip_preflight:
        run_preflight_subprocess(script_path=script_path, output_dir=output_dir)

    fit_records = []
    benchmark_records = []
    all_records = []

    if args.stage in {"fit", "all"}:
        fit_requests = expand_matrix(
            case_ids=args.case,
            gpu_counts=args.gpus,
            stage="fit",
            output_dir=output_dir,
            video_dir=video_dir,
            prompt=args.prompt,
            save_video=args.save_video,
            save_fit_video=args.save_fit_video,
            seed=args.seed,
            fastvideo_attention_backend=args.fastvideo_attention_backend,
            num_frames=args.num_frames,
        )
        for request in fit_requests:
            record = run_subprocess_request(
                request,
                script_path=script_path,
                timeout_seconds=args.timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
            )
            fit_records.append(record)
            all_records.append(record)
            if record["status"] != "passed" and not args.continue_on_error:
                write_summary(output_dir, all_records)
                return 1

    if args.stage == "benchmark":
        benchmark_case_ids = args.case
        benchmark_gpu_counts = args.gpus
    else:
        passed_fit_pairs = {
            (record["case_id"], int(record["gpus"]))
            for record in fit_records
            if record.get("status") == "passed"
        }
        benchmark_case_ids = sorted({case_id for case_id, _ in passed_fit_pairs})
        benchmark_gpu_counts = sorted({gpus for _, gpus in passed_fit_pairs})

    if args.stage in {"benchmark", "all"} and benchmark_case_ids and benchmark_gpu_counts:
        benchmark_requests = expand_matrix(
            case_ids=benchmark_case_ids,
            gpu_counts=benchmark_gpu_counts,
            stage="benchmark",
            output_dir=output_dir,
            video_dir=video_dir,
            prompt=args.prompt,
            save_video=args.save_video,
            seed=args.seed,
            fastvideo_attention_backend=args.fastvideo_attention_backend,
            num_frames=args.num_frames,
        )
        if args.stage == "all":
            benchmark_requests = [
                request
                for request in benchmark_requests
                if (request.case_id, request.gpus)
                in {
                    (record["case_id"], int(record["gpus"]))
                    for record in fit_records
                    if record.get("status") == "passed"
                }
            ]

        for request in benchmark_requests:
            record = run_subprocess_request(
                request,
                script_path=script_path,
                timeout_seconds=args.timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
            )
            benchmark_records.append(record)
            all_records.append(record)
            if record["status"] != "passed" and not args.continue_on_error:
                write_summary(output_dir, all_records)
                return 1

    summary_path = write_summary(output_dir, all_records)
    print(f"Wrote benchmark output to {output_dir}")
    print(f"Wrote summary to {summary_path}")
    return 0 if all(record.get("status") == "passed" for record in all_records) else 1


def main() -> int:
    args = parse_args()
    if args.worker_config is not None:
        return run_worker_request_file(args.worker_config)
    if args.worker_preflight is not None:
        run_preflight(args.worker_preflight)
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
