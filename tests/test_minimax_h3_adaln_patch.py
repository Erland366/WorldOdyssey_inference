from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_minimax_h3_adaln_patch_rebuilds_all_plans_after_overflow(
    tmp_path: Path,
) -> None:
    source = tmp_path / "minimax_h3.py"
    source.write_text(
        """        missing = {k: v for k, v in wanted.items() if k not in self._slots}
        if not missing:
            return
        if len(self._slots) + len(missing) > self.max_plans:
            # Every plan a request looks up has to stay resident for the whole
            # denoise loop, so an overflow means the capacity is too small --
            # evicting part of it would only move the failure into lookup().
            self._slots.clear()
            self.plan_lengths.zero_()
        if len(missing) > self.max_plans:
""",
        encoding="utf-8",
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "patch_minimax_h3_adaln_cache.py"
    )

    first = subprocess.run(
        [sys.executable, str(script), str(source)],
        check=True,
        text=True,
        capture_output=True,
    )
    patched = source.read_text(encoding="utf-8")
    second = subprocess.run(
        [sys.executable, str(script), str(source)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Applied" in first.stdout
    assert "already applied" in second.stdout
    assert source.read_text(encoding="utf-8") == patched
    invalidation = patched.index("self.plan_lengths.zero_()")
    rebuild = patched.index("missing = wanted")
    capacity_check = patched.index("if len(missing) > self.max_plans")
    assert invalidation < rebuild < capacity_check
