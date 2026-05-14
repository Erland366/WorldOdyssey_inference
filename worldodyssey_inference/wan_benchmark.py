from __future__ import annotations

import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Literal


BackendName = Literal["diffusers", "fastvideo"]
StageName = Literal["fit", "benchmark"]

DEFAULT_PROMPT = "A camera glides over a quiet city street at night, realistic, cinematic lighting."
DEFAULT_FASTVIDEO_ATTENTION_BACKEND = "VIDEO_SPARSE_ATTN"
DEFAULT_PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
DEFAULT_SEED = 1024
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


@dataclass(frozen=True)
class GenerationDefaults:
    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    fps: int


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    backend: BackendName
    model_id: str
    label: str
    defaults: GenerationDefaults


BENCHMARK_CASES: dict[str, BenchmarkCase] = {
    "diffusers-1.3b": BenchmarkCase(
        case_id="diffusers-1.3b",
        backend="diffusers",
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        label="Diffusers Wan2.1 T2V 1.3B",
        defaults=GenerationDefaults(
            height=480,
            width=832,
            num_frames=81,
            num_inference_steps=50,
            guidance_scale=5.0,
            fps=16,
        ),
    ),
    "fastvideo-1.3b": BenchmarkCase(
        case_id="fastvideo-1.3b",
        backend="fastvideo",
        model_id="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        label="FastVideo FastWan2.1 T2V 1.3B",
        defaults=GenerationDefaults(
            height=448,
            width=832,
            num_frames=61,
            num_inference_steps=3,
            guidance_scale=3.0,
            fps=16,
        ),
    ),
    "diffusers-5b": BenchmarkCase(
        case_id="diffusers-5b",
        backend="diffusers",
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        label="Diffusers Wan2.2 TI2V 5B",
        defaults=GenerationDefaults(
            height=704,
            width=1280,
            num_frames=121,
            num_inference_steps=50,
            guidance_scale=5.0,
            fps=24,
        ),
    ),
    "fastvideo-5b": BenchmarkCase(
        case_id="fastvideo-5b",
        backend="fastvideo",
        model_id="FastVideo/FastWan2.2-TI2V-5B-Diffusers",
        label="FastVideo FastWan2.2 TI2V 5B",
        defaults=GenerationDefaults(
            height=704,
            width=1280,
            num_frames=121,
            num_inference_steps=50,
            guidance_scale=5.0,
            fps=24,
        ),
    ),
}


@dataclass(frozen=True)
class RunRequest:
    case_id: str
    stage: StageName
    gpus: int
    output_dir: str
    video_dir: str
    prompt: str = DEFAULT_PROMPT
    save_video: bool = True
    seed: int = DEFAULT_SEED
    fastvideo_attention_backend: str = DEFAULT_FASTVIDEO_ATTENTION_BACKEND
    num_frames: int | None = None


class BenchmarkPhaseError(RuntimeError):
    def __init__(self, phase: str, traceback_text: str) -> None:
        super().__init__(f"Benchmark failed during {phase}")
        self.phase = phase
        self.traceback_text = traceback_text


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_output_dir(repo_root: Path) -> Path:
    return repo_root / "benchmark_results" / "wan_backend_benchmarks" / timestamp_slug()


def default_video_dir(repo_root: Path, output_dir: Path) -> Path:
    return repo_root / "artifacts" / "benchmark-videos" / output_dir.name


def resolve_cases(case_ids: list[str] | None) -> list[str]:
    if not case_ids:
        return list(BENCHMARK_CASES)
    unknown = sorted(set(case_ids) - set(BENCHMARK_CASES))
    if unknown:
        valid = ", ".join(BENCHMARK_CASES)
        raise ValueError(f"Unknown benchmark case(s): {', '.join(unknown)}. Valid cases: {valid}.")
    return case_ids


def stage_names(stage: str) -> list[StageName]:
    if stage == "all":
        return ["fit", "benchmark"]
    if stage in {"fit", "benchmark"}:
        return [stage]  # type: ignore[list-item]
    raise ValueError(f"Unknown benchmark stage {stage!r}.")


def run_id_for_request(request: RunRequest) -> str:
    return f"{request.stage}__{request.case_id}__{request.gpus}gpu"


def record_path_for_request(request: RunRequest) -> Path:
    return Path(request.output_dir) / "records" / f"{run_id_for_request(request)}.json"


def request_path_for_request(request: RunRequest) -> Path:
    return Path(request.output_dir) / "requests" / f"{run_id_for_request(request)}.json"


def log_path_for_request(request: RunRequest) -> Path:
    return Path(request.output_dir) / "logs" / f"{run_id_for_request(request)}.log"


def video_path_for_request(request: RunRequest) -> Path:
    return Path(request.video_dir) / f"{run_id_for_request(request)}.mp4"


def request_to_json(request: RunRequest) -> dict[str, Any]:
    return asdict(request)


def request_from_json(data: dict[str, Any]) -> RunRequest:
    return RunRequest(**data)


def expand_matrix(
    *,
    case_ids: list[str] | None,
    gpu_counts: list[int],
    stage: str,
    output_dir: Path,
    video_dir: Path,
    prompt: str = DEFAULT_PROMPT,
    save_video: bool = True,
    save_fit_video: bool = False,
    seed: int = DEFAULT_SEED,
    fastvideo_attention_backend: str = DEFAULT_FASTVIDEO_ATTENTION_BACKEND,
    num_frames: int | None = None,
) -> list[RunRequest]:
    cases = resolve_cases(case_ids)
    requests: list[RunRequest] = []
    for stage_name in stage_names(stage):
        stage_save_video = save_video and (stage_name == "benchmark" or save_fit_video)
        for case_id in cases:
            for gpus in gpu_counts:
                if gpus <= 0:
                    raise ValueError(f"GPU count must be positive, got {gpus}.")
                requests.append(
                    RunRequest(
                        case_id=case_id,
                        stage=stage_name,
                        gpus=gpus,
                        output_dir=str(output_dir),
                        video_dir=str(video_dir),
                        prompt=prompt,
                        save_video=stage_save_video,
                        seed=seed,
                        fastvideo_attention_backend=fastvideo_attention_backend,
                        num_frames=num_frames,
                    )
                )
    return requests


def select_cuda_visible_devices(gpus: int, base_env: dict[str, str]) -> str:
    existing = base_env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if existing and existing not in {"-1", "NoDevFiles"}:
        available = [item.strip() for item in existing.split(",") if item.strip()]
        if len(available) < gpus:
            raise ValueError(
                f"Requested {gpus} GPU(s), but CUDA_VISIBLE_DEVICES exposes only {len(available)}: {existing}."
            )
        return ",".join(available[:gpus])
    return ",".join(str(index) for index in range(gpus))


def parse_visible_gpu_indices(cuda_visible_devices: str) -> list[int]:
    indices: list[int] = []
    for item in cuda_visible_devices.split(","):
        item = item.strip()
        if item.isdecimal():
            indices.append(int(item))
    return indices


def prepare_child_environment(request: RunRequest, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env["CUDA_VISIBLE_DEVICES"] = select_cuda_visible_devices(request.gpus, env)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", DEFAULT_PYTORCH_CUDA_ALLOC_CONF)

    case = BENCHMARK_CASES[request.case_id]
    if case.backend == "fastvideo":
        env["CC"] = "/usr/bin/gcc"
        env["CXX"] = "/usr/bin/g++"
        env["FASTVIDEO_ATTENTION_BACKEND"] = request.fastvideo_attention_backend
        existing_path = env.get("PATH", "")
        env["PATH"] = f"/usr/bin:{existing_path}" if existing_path else "/usr/bin"

    return env


def generation_kwargs_for_request(request: RunRequest) -> dict[str, Any]:
    case = BENCHMARK_CASES[request.case_id]
    defaults = case.defaults
    num_inference_steps = 1 if request.stage == "fit" else defaults.num_inference_steps
    num_frames = request.num_frames if request.num_frames is not None else defaults.num_frames
    return {
        "height": defaults.height,
        "width": defaults.width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": defaults.guidance_scale,
    }


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in (
        "torch",
        "diffusers",
        "fastvideo",
        "accelerate",
        "transformers",
        "imageio",
        "imageio-ffmpeg",
    ):
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "not installed"
    return versions


def run_command(command: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)


def gpu_inventory() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []

    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        return []

    inventory: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        inventory.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(parts[2]),
                "driver_version": parts[3],
            }
        )
    return inventory


def gpu_memory_used_mib(gpu_indices: list[int]) -> dict[str, int]:
    if not gpu_indices or shutil.which("nvidia-smi") is None:
        return {}

    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    if result.returncode != 0:
        return {}

    wanted = set(gpu_indices)
    used: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        index = int(parts[0])
        if index in wanted:
            used[str(index)] = int(parts[1])
    return used


def update_peak_memory(peak: dict[str, int], sample: dict[str, int]) -> None:
    for index, used_mib in sample.items():
        peak[index] = max(peak.get(index, 0), used_mib)


def classify_failure(text: str, phase: str | None = None) -> str:
    lowered = text.lower()
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "oom"
    if (
        "driver version is insufficient" in lowered
        or "unsupported display driver" in lowered
        or "cuda runtime version" in lowered
        or "named symbol not found" in lowered
    ):
        return "cuda_runtime_mismatch"
    if "importerror" in lowered or "modulenotfounderror" in lowered or "no module named" in lowered:
        return "import_error"
    if phase == "load":
        return "load_error"
    if phase == "generation":
        return "generation_error"
    return "error"


def base_record(request: RunRequest) -> dict[str, Any]:
    case = BENCHMARK_CASES[request.case_id]
    return {
        "schema_version": 1,
        "run_id": run_id_for_request(request),
        "created_at": utc_now_iso(),
        "case_id": request.case_id,
        "case_label": case.label,
        "backend": case.backend,
        "model_id": case.model_id,
        "stage": request.stage,
        "gpus": request.gpus,
        "prompt": request.prompt,
        "seed": request.seed,
        "save_video": request.save_video,
        "generation": generation_kwargs_for_request(request),
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "gpus": gpu_inventory(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "fastvideo_attention_backend": os.environ.get("FASTVIDEO_ATTENTION_BACKEND"),
        },
        "status": "started",
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_synchronize_if_available() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_diffusers_case(request: RunRequest) -> dict[str, Any]:
    phase = "import"
    try:
        import torch
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video

        case = BENCHMARK_CASES[request.case_id]
        generation_kwargs = generation_kwargs_for_request(request)

        set_reproducibility(request.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        phase = "load"
        load_start = time.perf_counter()
        torch_dtype = {
            "default": torch.bfloat16,
            "vae": torch.float32,
        }
        if request.gpus == 1:
            pipe = WanPipeline.from_pretrained(case.model_id, torch_dtype=torch_dtype)
            pipe.to("cuda")
        else:
            pipe = WanPipeline.from_pretrained(
                case.model_id,
                torch_dtype=torch_dtype,
                device_map="balanced",
            )
        pipe.set_progress_bar_config(disable=True)
        cuda_synchronize_if_available()
        load_time_s = time.perf_counter() - load_start

        generator = torch.Generator(device="cuda").manual_seed(request.seed)
        run_kwargs = {
            "prompt": request.prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "generator": generator,
            **generation_kwargs,
        }

        phase = "generation"
        generation_start = time.perf_counter()
        frames = pipe(**run_kwargs).frames[0]
        cuda_synchronize_if_available()
        generation_time_s = time.perf_counter() - generation_start

        output_video_path: str | None = None
        if request.save_video:
            phase = "export"
            output_path = video_path_for_request(request)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(output_path), fps=case.defaults.fps)
            output_video_path = str(output_path)

        return {
            "load_time_s": load_time_s,
            "generation_time_s": generation_time_s,
            "total_time_s": load_time_s + generation_time_s,
            "output_video_path": output_video_path,
        }
    except Exception as exc:
        raise BenchmarkPhaseError(
            phase,
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        ) from exc


def run_fastvideo_case(request: RunRequest) -> dict[str, Any]:
    phase = "import"
    try:
        import torch

        from worldodyssey_inference.fastvideo_compat import configure_fastvideo_torch_compat

        configure_fastvideo_torch_compat()
        from fastvideo import VideoGenerator

        case = BENCHMARK_CASES[request.case_id]
        generation_kwargs = generation_kwargs_for_request(request)

        set_reproducibility(request.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        phase = "load"
        load_start = time.perf_counter()
        generator = VideoGenerator.from_pretrained(
            case.model_id,
            num_gpus=request.gpus,
        )
        cuda_synchronize_if_available()
        load_time_s = time.perf_counter() - load_start

        output_path = video_path_for_request(request)
        if request.save_video:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        phase = "generation"
        generation_start = time.perf_counter()
        try:
            generator.generate_video(
                request.prompt,
                output_path=str(output_path),
                save_video=request.save_video,
                **generation_kwargs,
            )
            cuda_synchronize_if_available()
        finally:
            generator.shutdown()
        generation_time_s = time.perf_counter() - generation_start

        return {
            "load_time_s": load_time_s,
            "generation_time_s": generation_time_s,
            "total_time_s": load_time_s + generation_time_s,
            "output_video_path": str(output_path) if request.save_video else None,
        }
    except Exception as exc:
        raise BenchmarkPhaseError(
            phase,
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        ) from exc


def run_worker_request(request: RunRequest) -> dict[str, Any]:
    case = BENCHMARK_CASES[request.case_id]
    record = base_record(request)
    phase = "load"
    started = time.perf_counter()

    try:
        if case.backend == "diffusers":
            metrics = run_diffusers_case(request)
        elif case.backend == "fastvideo":
            metrics = run_fastvideo_case(request)
        else:
            raise ValueError(f"Unsupported backend {case.backend!r}.")
        record.update(metrics)
        record["status"] = "passed"
    except BenchmarkPhaseError as exc:
        record["status"] = classify_failure(exc.traceback_text, phase=exc.phase)
        record["phase"] = exc.phase
        record["error"] = exc.traceback_text
    except Exception as exc:
        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        record["status"] = classify_failure(error_text, phase=phase)
        record["error"] = error_text
    finally:
        record["worker_wall_time_s"] = time.perf_counter() - started

    return record


def run_worker_request_file(request_path: Path) -> int:
    request = request_from_json(json.loads(request_path.read_text(encoding="utf-8")))
    result = run_worker_request(request)
    write_json(record_path_for_request(request), result)
    return 0 if result["status"] == "passed" else 1


def tail_text(path: Path, *, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def make_missing_result_record(
    request: RunRequest,
    *,
    status: str,
    return_code: int | None,
    timed_out: bool,
    log_path: Path,
) -> dict[str, Any]:
    record = base_record(request)
    record["status"] = status
    record["return_code"] = return_code
    record["timed_out"] = timed_out
    record["log_path"] = str(log_path)
    record["error"] = tail_text(log_path)
    return record


def run_subprocess_request(
    request: RunRequest,
    *,
    script_path: Path,
    timeout_seconds: int,
    poll_interval_seconds: float,
    base_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for child_dir in ("requests", "records", "logs"):
        (output_dir / child_dir).mkdir(parents=True, exist_ok=True)

    request_path = request_path_for_request(request)
    result_path = record_path_for_request(request)
    log_path = log_path_for_request(request)
    write_json(request_path, request_to_json(request))

    env = prepare_child_environment(request, base_env=base_env)
    visible_gpu_indices = parse_visible_gpu_indices(env["CUDA_VISIBLE_DEVICES"])
    peak_memory = {str(index): 0 for index in visible_gpu_indices}
    command = [sys.executable, str(script_path), "--worker-config", str(request_path)]

    started = time.perf_counter()
    timed_out = False
    return_code: int | None = None
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(script_path.resolve().parents[1]),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            update_peak_memory(peak_memory, gpu_memory_used_mib(visible_gpu_indices))
            if time.perf_counter() - started > timeout_seconds:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                break
            time.sleep(poll_interval_seconds)

        update_peak_memory(peak_memory, gpu_memory_used_mib(visible_gpu_indices))
        return_code = process.poll()

    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        status = "timeout" if timed_out else classify_failure(tail_text(log_path))
        result = make_missing_result_record(
            request,
            status=status,
            return_code=return_code,
            timed_out=timed_out,
            log_path=log_path,
        )

    result["command"] = command
    result["return_code"] = return_code
    result["timed_out"] = timed_out
    result["log_path"] = str(log_path)
    result["peak_vram_mib_by_gpu"] = peak_memory
    result["peak_vram_mib_total"] = sum(peak_memory.values())
    result["parent_wall_time_s"] = time.perf_counter() - started
    result["env"]["cuda_visible_devices"] = env["CUDA_VISIBLE_DEVICES"]
    result["env"]["important_env_vars"] = {
        key: env[key]
        for key in ("CUDA_VISIBLE_DEVICES", "CC", "CXX", "FASTVIDEO_ATTENTION_BACKEND", "PYTORCH_CUDA_ALLOC_CONF")
        if key in env
    }

    write_json(result_path, result)
    append_jsonl(output_dir / "results.jsonl", result)
    return result


def run_preflight(output_path: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, value: Any = None, error: str | None = None) -> None:
        check = {"name": name, "status": "passed" if error is None else "failed"}
        if value is not None:
            check["value"] = value
        if error is not None:
            check["error"] = error
        checks.append(check)

    add_check("packages", package_versions())
    add_check("gpu_inventory", gpu_inventory())

    try:
        import torch

        value = {
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            x = torch.randn(1, 4, 5, 8, 8, device="cuda", dtype=torch.bfloat16)
            conv = torch.nn.Conv3d(4, 8, 3, padding=1).cuda().bfloat16()
            with torch.no_grad():
                y = conv(x)
            torch.cuda.synchronize()
            value["conv3d_shape"] = tuple(y.shape)
        add_check("torch_cuda_probe", value)
    except Exception as exc:
        add_check("torch_cuda_probe", error="".join(traceback.format_exception_only(type(exc), exc)).strip())

    try:
        from diffusers import WanPipeline

        add_check("diffusers_wan_import", str(WanPipeline))
    except Exception as exc:
        add_check("diffusers_wan_import", error="".join(traceback.format_exception_only(type(exc), exc)).strip())

    try:
        from worldodyssey_inference.fastvideo_compat import configure_fastvideo_torch_compat

        configure_fastvideo_torch_compat()
        from fastvideo import VideoGenerator

        add_check("fastvideo_import", str(VideoGenerator))
    except Exception as exc:
        add_check("fastvideo_import", error="".join(traceback.format_exception_only(type(exc), exc)).strip())

    record = {
        "created_at": utc_now_iso(),
        "status": "passed" if all(check["status"] == "passed" for check in checks) else "failed",
        "checks": checks,
    }
    write_json(output_path, record)
    return record


def run_preflight_subprocess(
    *,
    script_path: Path,
    output_dir: Path,
    timeout_seconds: int = 180,
    base_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = output_dir / "preflight.json"
    log_path = output_dir / "logs" / "preflight.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(base_env or os.environ)
    env["CUDA_VISIBLE_DEVICES"] = select_cuda_visible_devices(1, env)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", DEFAULT_PYTORCH_CUDA_ALLOC_CONF)
    env["CC"] = "/usr/bin/gcc"
    env["CXX"] = "/usr/bin/g++"
    env["FASTVIDEO_ATTENTION_BACKEND"] = DEFAULT_FASTVIDEO_ATTENTION_BACKEND
    env["PATH"] = f"/usr/bin:{env.get('PATH', '')}"

    command = [sys.executable, str(script_path), "--worker-preflight", str(preflight_path)]
    with log_path.open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            cwd=str(script_path.resolve().parents[1]),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

    if preflight_path.exists():
        record = json.loads(preflight_path.read_text(encoding="utf-8"))
    else:
        record = {
            "created_at": utc_now_iso(),
            "status": "failed",
            "error": tail_text(log_path),
        }
        write_json(preflight_path, record)

    record["return_code"] = result.returncode
    record["log_path"] = str(log_path)
    write_json(preflight_path, record)
    return record


def result_row(record: dict[str, Any]) -> str:
    peak_by_gpu = record.get("peak_vram_mib_by_gpu", {})
    if peak_by_gpu:
        peak_vram = ", ".join(f"GPU {gpu}: {mib} MiB" for gpu, mib in sorted(peak_by_gpu.items()))
    else:
        peak_vram = "n/a"

    return (
        f"| {record.get('stage', '')} | {record.get('case_id', '')} | {record.get('backend', '')} | "
        f"{record.get('gpus', '')} | {record.get('status', '')} | "
        f"{record.get('load_time_s', 0):.2f} | {record.get('generation_time_s', 0):.2f} | "
        f"{peak_vram} | {record.get('output_video_path') or ''} |"
    )


def render_summary(records: list[dict[str, Any]], *, preflight: dict[str, Any] | None = None) -> str:
    lines = [
        "# Wan Backend Benchmark Summary",
        "",
        f"Generated at: {utc_now_iso()}",
        "",
    ]
    if preflight is not None:
        lines.extend(
            [
                "## Preflight",
                "",
                f"Status: `{preflight.get('status', 'unknown')}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Results",
            "",
            "| Stage | Case | Backend | GPUs | Status | Load s | Generate s | Peak VRAM | Video |",
            "|---|---|---:|---:|---|---:|---:|---|---|",
        ]
    )
    lines.extend(result_row(record) for record in records)
    lines.append("")
    return "\n".join(lines)


def write_summary(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    preflight_path = output_dir / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.exists() else None
    summary_path = output_dir / "summary.md"
    summary_path.write_text(render_summary(records, preflight=preflight), encoding="utf-8")
    return summary_path
