from __future__ import annotations

import os
from pathlib import Path


def configure_fastvideo_torch_compat() -> None:
    """Apply local compatibility shims needed before importing FastVideo."""
    import torch._dynamo.config as dynamo_config

    repo_root = str(Path(__file__).resolve().parents[1])
    pythonpath = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in pythonpath.split(os.pathsep) if entry]
    if repo_root not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_root, *entries])

    config = getattr(dynamo_config, "_config", None)
    if isinstance(config, dict) and "recompile_limit" not in config:
        config["recompile_limit"] = config.get("cache_size_limit", 8)
