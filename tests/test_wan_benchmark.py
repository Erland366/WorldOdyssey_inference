from __future__ import annotations

from pathlib import Path

from worldodyssey_inference.wan_benchmark import (
    DEFAULT_FASTVIDEO_ATTENTION_BACKEND,
    DEFAULT_PYTORCH_CUDA_ALLOC_CONF,
    RunRequest,
    classify_failure,
    expand_matrix,
    generation_kwargs_for_request,
    prepare_child_environment,
    render_summary,
)


def make_request(case_id: str, gpus: int = 1, stage: str = "fit") -> RunRequest:
    return RunRequest(
        case_id=case_id,
        stage=stage,  # type: ignore[arg-type]
        gpus=gpus,
        output_dir="/tmp/bench-output",
        video_dir="/tmp/bench-videos",
    )


def test_expand_matrix_all_cases_and_gpu_counts(tmp_path: Path) -> None:
    requests = expand_matrix(
        case_ids=None,
        gpu_counts=[1, 2],
        stage="fit",
        output_dir=tmp_path / "results",
        video_dir=tmp_path / "videos",
    )

    assert len(requests) == 8
    assert {(request.case_id, request.gpus) for request in requests} == {
        ("diffusers-1.3b", 1),
        ("diffusers-1.3b", 2),
        ("fastvideo-1.3b", 1),
        ("fastvideo-1.3b", 2),
        ("diffusers-5b", 1),
        ("diffusers-5b", 2),
        ("fastvideo-5b", 1),
        ("fastvideo-5b", 2),
    }
    assert all(not request.save_video for request in requests)


def test_expand_matrix_all_stage_disables_fit_videos_only(tmp_path: Path) -> None:
    requests = expand_matrix(
        case_ids=["fastvideo-5b"],
        gpu_counts=[1],
        stage="all",
        output_dir=tmp_path / "results",
        video_dir=tmp_path / "videos",
    )

    assert [(request.stage, request.save_video) for request in requests] == [
        ("fit", False),
        ("benchmark", True),
    ]


def test_expand_matrix_can_save_fit_videos(tmp_path: Path) -> None:
    requests = expand_matrix(
        case_ids=["fastvideo-5b"],
        gpu_counts=[2],
        stage="fit",
        output_dir=tmp_path / "results",
        video_dir=tmp_path / "videos",
        save_fit_video=True,
    )

    assert [(request.stage, request.save_video) for request in requests] == [("fit", True)]


def test_diffusers_single_gpu_environment_uses_one_visible_device() -> None:
    env = prepare_child_environment(
        make_request("diffusers-1.3b", gpus=1),
        base_env={"PATH": "/venv/bin:/bin"},
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == DEFAULT_PYTORCH_CUDA_ALLOC_CONF
    assert "FASTVIDEO_ATTENTION_BACKEND" not in env
    assert env["PATH"] == "/venv/bin:/bin"


def test_diffusers_two_gpu_environment_respects_existing_visible_devices() -> None:
    env = prepare_child_environment(
        make_request("diffusers-5b", gpus=2),
        base_env={"CUDA_VISIBLE_DEVICES": "2,3,4", "PATH": "/venv/bin:/bin"},
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == DEFAULT_PYTORCH_CUDA_ALLOC_CONF


def test_fastvideo_environment_sets_compilers_and_sparse_attention() -> None:
    env = prepare_child_environment(
        make_request("fastvideo-5b", gpus=2),
        base_env={"PATH": "/venv/bin:/home/coder/miniconda3/bin:/bin"},
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["CC"] == "/usr/bin/gcc"
    assert env["CXX"] == "/usr/bin/g++"
    assert env["FASTVIDEO_ATTENTION_BACKEND"] == DEFAULT_FASTVIDEO_ATTENTION_BACKEND
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == DEFAULT_PYTORCH_CUDA_ALLOC_CONF
    assert env["PATH"].startswith("/usr/bin:")


def test_child_environment_respects_existing_allocator_config() -> None:
    env = prepare_child_environment(
        make_request("diffusers-1.3b", gpus=1),
        base_env={"PATH": "/venv/bin:/bin", "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256"},
    )

    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:256"


def test_fit_stage_overrides_steps_only() -> None:
    fit_kwargs = generation_kwargs_for_request(make_request("fastvideo-5b", stage="fit"))
    benchmark_kwargs = generation_kwargs_for_request(make_request("fastvideo-5b", stage="benchmark"))

    assert fit_kwargs["num_inference_steps"] == 1
    assert fit_kwargs["height"] == benchmark_kwargs["height"] == 704
    assert fit_kwargs["width"] == benchmark_kwargs["width"] == 1280
    assert fit_kwargs["num_frames"] == benchmark_kwargs["num_frames"] == 121
    assert benchmark_kwargs["num_inference_steps"] == 50


def test_num_frames_override_applies_to_fit_and_benchmark(tmp_path: Path) -> None:
    requests = expand_matrix(
        case_ids=["diffusers-1.3b"],
        gpu_counts=[2],
        stage="all",
        output_dir=tmp_path / "results",
        video_dir=tmp_path / "videos",
        num_frames=121,
    )

    assert [generation_kwargs_for_request(request)["num_frames"] for request in requests] == [121, 121]


def test_classify_common_failures() -> None:
    assert classify_failure("RuntimeError: CUDA out of memory.") == "oom"
    assert classify_failure("Cuda failure 'CUDA driver version is insufficient for CUDA runtime version'") == (
        "cuda_runtime_mismatch"
    )
    assert classify_failure("ModuleNotFoundError: No module named 'fastvideo'") == "import_error"
    assert classify_failure("ValueError: checkpoint missing", phase="load") == "load_error"
    assert classify_failure("RuntimeError: failed later", phase="generation") == "generation_error"


def test_render_summary_includes_core_metrics() -> None:
    summary = render_summary(
        [
            {
                "stage": "fit",
                "case_id": "fastvideo-5b",
                "backend": "fastvideo",
                "gpus": 1,
                "status": "passed",
                "load_time_s": 10.2,
                "generation_time_s": 20.4,
                "peak_vram_mib_by_gpu": {"0": 23123},
                "output_video_path": "",
            }
        ],
        preflight={"status": "passed"},
    )

    assert "Status: `passed`" in summary
    assert "| fit | fastvideo-5b | fastvideo | 1 | passed | 10.20 | 20.40 | GPU 0: 23123 MiB |  |" in summary
