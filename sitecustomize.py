"""Local interpreter startup compatibility hooks for this project."""

from __future__ import annotations

from worldodyssey_inference.fastvideo_compat import configure_fastvideo_torch_compat

configure_fastvideo_torch_compat()
