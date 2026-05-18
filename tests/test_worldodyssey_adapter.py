from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from scripts.submit_video_batch import build_batch_submission_plan, parse_args as parse_batch_args
from scripts.submit_worldodyssey_task import build_submission_plan, parse_args
from worldodyssey_inference.video_backend.models import VideoMode
from worldodyssey_inference.video_backend.providers import (
    DEBUG_TINY_WAN_T2V_MODEL,
    DEFAULT_SGLANG_I2V_MODEL,
    DEFAULT_SGLANG_MODEL,
)
from worldodyssey_inference.video_backend.worldodyssey import (
    DEFAULT_WORLDODYSSEY_TASK_DIR,
    build_worldodyssey_generation_request,
    build_worldodyssey_prompt,
    load_worldodyssey_task,
)


def make_task_dir(
    tmp_path: Path,
    *,
    name: str = "move_cup",
    task_text: str = "Move the cup from the table to the shelf.",
) -> Path:
    task_dir = tmp_path / name
    frames_dir = task_dir / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "main.png").write_bytes(b"main-frame")
    (frames_dir / "extra.png").write_bytes(b"extra-frame")
    (task_dir / "gt.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task": task_text,
                "frames": {
                    "main": "frames/main.png",
                    "extras": ["frames/extra.png"],
                },
                "topology_graph": {
                    "states": {
                        "A": "The cup is on the table.",
                        "B": "The cup is in hand.",
                        "C": "The cup is on the shelf.",
                    },
                    "edges": {
                        "A,B": {
                            "action": "Pick up the cup from the table.",
                            "objects": ["cup", "table", "hand"],
                            "constraints": ["The cup must not disappear during pickup."],
                        },
                        "B,C": {
                            "action": "Place the cup on the shelf.",
                            "objects": ["cup", "shelf", "hand"],
                            "constraints": ["The cup must end resting on the shelf."],
                        },
                        "A,C": {
                            "action": "Carry the cup directly only if the task setup requires it.",
                            "objects": ["cup"],
                            "constraints": ["Do not use this shortcut if it skips required visible contact."],
                        },
                    },
                    "valid_topo_sorts": [["A", "B", "C"]],
                    "start_state": "A",
                    "goal_state": "C",
                },
            }
        ),
        encoding="utf-8",
    )
    return task_dir


def test_worldodyssey_default_task_dir_uses_submodule() -> None:
    expected_suffix = Path("submodule/worldodyssey/inputs/move_bookmark")

    assert (
        DEFAULT_WORLDODYSSEY_TASK_DIR.parts[-len(expected_suffix.parts) :]
        == expected_suffix.parts
    )


def test_worldodyssey_prompt_uses_task_text_only(tmp_path: Path) -> None:
    loaded_task = load_worldodyssey_task(make_task_dir(tmp_path))

    prompt = build_worldodyssey_prompt(loaded_task)

    assert prompt == "Move the cup from the table to the shelf."
    assert "Start state" not in prompt
    assert "Pick up the cup from the table." not in prompt


def test_worldodyssey_prompt_can_prepend_config_text(tmp_path: Path) -> None:
    loaded_task = load_worldodyssey_task(make_task_dir(tmp_path))

    prompt = build_worldodyssey_prompt(
        loaded_task,
        prompt_prefix="Generate an egocentric first-person video.",
    )

    assert prompt == "Generate an egocentric first-person video.\nMove the cup from the table to the shelf."


def test_worldodyssey_request_targets_local_sglang_text_to_video(tmp_path: Path) -> None:
    loaded_task = load_worldodyssey_task(make_task_dir(tmp_path))

    request = build_worldodyssey_generation_request(loaded_task)

    assert request.provider == "sglang"
    assert request.model == DEFAULT_SGLANG_MODEL
    assert request.mode == VideoMode.TEXT_TO_VIDEO
    assert request.prompt == "Move the cup from the table to the shelf."
    assert request.image_base64 is None
    assert request.options.height == 448
    assert request.options.width == 832
    assert request.options.num_frames == 61
    assert request.options.num_inference_steps is None
    assert request.options.seed is None
    assert request.options.attention_backend is None
    assert request.options.vsa_sparsity is None
    assert request.metadata["adapter"] == "worldodyssey"
    assert request.metadata["task_id"] == "move_cup"
    assert request.metadata["source_video_paths"]["gt"].endswith("gt.mp4")


def test_worldodyssey_tiny_t2v_request_does_not_inject_vsa_options(tmp_path: Path) -> None:
    loaded_task = load_worldodyssey_task(make_task_dir(tmp_path))

    request = build_worldodyssey_generation_request(
        loaded_task,
        model=DEBUG_TINY_WAN_T2V_MODEL,
        height=64,
        width=64,
        num_frames=5,
        num_inference_steps=1,
    )

    assert request.model == DEBUG_TINY_WAN_T2V_MODEL
    assert request.options.height == 64
    assert request.options.width == 64
    assert request.options.num_frames == 5
    assert request.options.num_inference_steps == 1
    assert request.options.attention_backend is None
    assert request.options.vsa_sparsity is None


def test_worldodyssey_request_can_attach_main_image_for_future_providers(tmp_path: Path) -> None:
    loaded_task = load_worldodyssey_task(make_task_dir(tmp_path))

    request = build_worldodyssey_generation_request(
        loaded_task,
        mode=VideoMode.IMAGE_TO_VIDEO,
        include_main_image_base64=True,
    )

    assert request.mode == VideoMode.IMAGE_TO_VIDEO
    assert request.model == DEFAULT_SGLANG_I2V_MODEL
    assert request.image_base64 == base64.b64encode(b"main-frame").decode("ascii")
    assert request.options.height == 480
    assert request.options.width == 832
    assert request.options.num_frames == 81
    assert request.options.num_inference_steps is None
    assert request.options.attention_backend is None
    assert request.options.vsa_sparsity is None


def test_worldodyssey_cli_keeps_mode_specific_defaults_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["submit_worldodyssey_task.py", "--i2v"])

    args = parse_args()

    assert args.i2v is True
    assert args.height is None
    assert args.width is None
    assert args.num_frames is None
    assert args.num_inference_steps is None
    assert args.attention_backend is None
    assert args.vsa_sparsity is None


def test_worldodyssey_yaml_config_can_submit_i2v_and_override_values(tmp_path: Path) -> None:
    task_dir = make_task_dir(tmp_path)
    config_path = tmp_path / "submit.yaml"
    config_path.write_text(
        f"""
task: {task_dir}
backend_url: http://example.test:8000
adapter:
  i2v: true
  prompt_prefix: Generate an egocentric first-person video.
request:
  provider: sglang
  options:
    height: 256
    num_frames: 17
run:
  dry_run: true
  wait: true
  download_path: artifacts/from-config.mp4
""",
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--set",
            "request.options.num_frames=9",
            "--set",
            "run.wait=false",
        ]
    )
    plan = build_submission_plan(args)

    assert plan.backend_url == "http://example.test:8000"
    assert plan.dry_run is True
    assert plan.wait is False
    assert plan.download_path == Path("artifacts/from-config.mp4")
    assert plan.request.mode == VideoMode.IMAGE_TO_VIDEO
    assert plan.request.model == DEFAULT_SGLANG_I2V_MODEL
    assert plan.request.prompt == "Generate an egocentric first-person video.\nMove the cup from the table to the shelf."
    assert plan.request.image_base64 == base64.b64encode(b"main-frame").decode("ascii")
    assert plan.request.options.height == 256
    assert plan.request.options.num_frames == 9


def test_worldodyssey_prompt_prefix_cli_flag_overrides_yaml_config(tmp_path: Path) -> None:
    task_dir = make_task_dir(tmp_path)
    config_path = tmp_path / "submit.yaml"
    config_path.write_text(
        f"""
task: {task_dir}
adapter:
  prompt_prefix: YAML prefix.
run:
  dry_run: true
""",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path), "--prompt-prefix", "CLI prefix."])
    plan = build_submission_plan(args)

    assert plan.request.prompt == "CLI prefix.\nMove the cup from the table to the shelf."


def test_worldodyssey_cli_flags_override_yaml_config(tmp_path: Path) -> None:
    task_dir = make_task_dir(tmp_path)
    config_path = tmp_path / "submit.yaml"
    config_path.write_text(
        f"""
task: {task_dir}
request:
  mode: text_to_video
  options:
    height: 480
    width: 832
run:
  dry_run: false
""",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path), "--height", "256", "--dry-run"])
    plan = build_submission_plan(args)

    assert plan.dry_run is True
    assert plan.request.options.height == 256
    assert plan.request.options.width == 832


def test_worldodyssey_parent_input_directory_builds_batch(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "inputs"
    make_task_dir(inputs_dir, name="move_cup")
    make_task_dir(
        inputs_dir,
        name="wipe_table",
        task_text="Wipe the spilled water from the table.",
    )
    skipped_dir = inputs_dir / "placeholder"
    skipped_dir.mkdir()
    (skipped_dir / ".gitkeep").write_text("", encoding="utf-8")
    config_path = tmp_path / "submit.yaml"
    config_path.write_text(
        f"""
task: {inputs_dir}
backend_url: http://example.test:8000
adapter:
  prompt_prefix: Generate an egocentric first-person video.
request:
  provider: sglang
  mode: text_to_video
  options:
    height: 448
    width: 832
    num_frames: 61
run:
  dry_run: true
  wait: true
  download_dir: artifacts/worldodyssey-batch
""",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path)])
    plan = build_submission_plan(args)

    assert plan.request is None
    assert plan.batch_request is not None
    assert plan.download_dir == Path("artifacts/worldodyssey-batch")
    assert plan.batch_request.metadata["task_root"] == str(inputs_dir.resolve())
    assert plan.batch_request.metadata["task_ids"] == ["move_cup", "wipe_table"]
    assert plan.batch_request.metadata["skipped_entries"] == [str(skipped_dir.resolve())]
    assert len(plan.batch_request.requests) == 2
    assert plan.batch_request.requests[0].metadata["task_id"] == "move_cup"
    assert plan.batch_request.requests[1].metadata["task_id"] == "wipe_table"
    assert plan.batch_request.requests[0].prompt.startswith("Generate an egocentric first-person video.")
    assert plan.batch_request.requests[1].prompt.endswith("Wipe the spilled water from the table.")


def test_worldodyssey_parent_tiny_t2v_config_does_not_inject_vsa_options(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "inputs"
    make_task_dir(inputs_dir, name="move_cup")
    config_path = tmp_path / "submit.yaml"
    config_path.write_text(
        f"""
task: {inputs_dir}
request:
  provider: sglang
  mode: text_to_video
  model: {DEBUG_TINY_WAN_T2V_MODEL}
  options:
    height: 64
    width: 64
    num_frames: 5
run:
  dry_run: true
  download_dir: artifacts/worldodyssey-tiny-batch
""",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path)])
    plan = build_submission_plan(args)

    assert plan.batch_request is not None
    request = plan.batch_request.requests[0]
    assert request.model == DEBUG_TINY_WAN_T2V_MODEL
    assert request.options.attention_backend is None
    assert request.options.vsa_sparsity is None


def test_worldodyssey_parent_input_directory_rejects_file_download_path(tmp_path: Path) -> None:
    inputs_dir = tmp_path / "inputs"
    make_task_dir(inputs_dir)
    config_path = tmp_path / "submit.yaml"
    config_path.write_text(
        f"""
task: {inputs_dir}
run:
  dry_run: true
  download_path: artifacts/single-file.mp4
""",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path)])

    with pytest.raises(ValueError, match="download_dir"):
        build_submission_plan(args)


def test_worldodyssey_single_task_rejects_batch_download_dir(tmp_path: Path) -> None:
    task_dir = make_task_dir(tmp_path)
    config_path = tmp_path / "submit.yaml"
    config_path.write_text(
        f"""
task: {task_dir}
run:
  dry_run: true
  download_dir: artifacts/batch-dir
""",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config_path)])

    with pytest.raises(ValueError, match="download_path"):
        build_submission_plan(args)


def test_video_batch_yaml_config_can_override_list_items(tmp_path: Path) -> None:
    config_path = tmp_path / "tiny-batch.yaml"
    config_path.write_text(
        f"""
backend_url: http://example.test:8000
requests:
  - provider: sglang
    model: {DEBUG_TINY_WAN_T2V_MODEL}
    mode: text_to_video
    prompt: first prompt
    options:
      height: 64
      width: 64
      num_frames: 5
      timeout_seconds: 300
  - provider: sglang
    model: {DEBUG_TINY_WAN_T2V_MODEL}
    mode: text_to_video
    prompt: second prompt
    options:
      height: 64
      width: 64
      num_frames: 5
      timeout_seconds: 300
metadata:
  purpose: tiny-debug
run:
  dry_run: true
  wait: false
""",
        encoding="utf-8",
    )

    args = parse_batch_args(
        [
            str(config_path),
            "--set",
            "requests.0.prompt=overridden prompt",
            "--set",
            "requests.1.options.timeout_seconds=789",
            "--wait",
        ]
    )
    plan = build_batch_submission_plan(args)

    assert plan.backend_url == "http://example.test:8000"
    assert plan.dry_run is True
    assert plan.wait is True
    assert plan.request.metadata == {"purpose": "tiny-debug"}
    assert len(plan.request.requests) == 2
    assert plan.request.requests[0].prompt == "overridden prompt"
    assert plan.request.requests[0].model == DEBUG_TINY_WAN_T2V_MODEL
    assert plan.request.requests[1].prompt == "second prompt"
    assert plan.request.requests[1].options.timeout_seconds == 789


def test_worldodyssey_loader_fails_fast_on_missing_frame(tmp_path: Path) -> None:
    task_dir = make_task_dir(tmp_path)
    (task_dir / "frames" / "main.png").unlink()

    with pytest.raises(FileNotFoundError, match="frame path"):
        load_worldodyssey_task(task_dir)
