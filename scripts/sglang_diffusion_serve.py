#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys

from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
    add_multimodal_gen_serve_args,
)
from sglang.multimodal_gen.runtime.launch_server import launch_server
from sglang.multimodal_gen.runtime.server_args import (
    ExecutionMode,
    ServerArgs,
    WorkloadType,
)


def patch_sglang_video_output_filename() -> None:
    """Patch SGLang 0.5.x video serving so /v1/videos can create output names."""
    from sglang.multimodal_gen.configs.sample.base import SamplingParams

    if getattr(SamplingParams.log, "_worldodyssey_output_filename_patch", False):
        return

    original_log = SamplingParams.log

    def log_with_output_filename(self: SamplingParams, server_args: ServerArgs) -> None:
        if self.output_file_name is None:
            self.set_output_file_name()
        original_log(self, server_args)

    log_with_output_filename._worldodyssey_output_filename_patch = True  # type: ignore[attr-defined]
    SamplingParams.log = log_with_output_filename  # type: ignore[method-assign]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the native SGLang Diffusion server.")
    return add_multimodal_gen_serve_args(parser)


def provided_arg_names(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_dests = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    names: set[str] = set()
    for arg in argv:
        if not arg.startswith("--"):
            continue
        option = arg.split("=", 1)[0]
        dest = option_dests.get(option)
        if dest is not None:
            names.add(dest)
    return names


def build_server_args(argv: list[str] | None = None) -> ServerArgs:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    parsed_args, unknown_args = parser.parse_known_args(argv)
    provided_names = provided_arg_names(parser, [*argv, *unknown_args])
    provided_args = {
        key: value
        for key, value in vars(parsed_args).items()
        if key in provided_names
    }

    if "mode" in provided_args and isinstance(provided_args["mode"], str):
        provided_args["mode"] = ExecutionMode.from_string(provided_args["mode"])
    if "workload_type" in provided_args and isinstance(provided_args["workload_type"], str):
        provided_args["workload_type"] = WorkloadType.from_string(provided_args["workload_type"])

    server_args = ServerArgs.from_dict(provided_args)
    server_args.post_init_serve()
    return server_args


def main(argv: list[str] | None = None) -> int:
    patch_sglang_video_output_filename()
    server_args = build_server_args(argv)
    launch_server(server_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
