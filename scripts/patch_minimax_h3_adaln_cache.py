#!/usr/bin/env python3
"""Apply the SGLang 0.5.18 MiniMax-H3 online AdaLN overflow fix."""

from pathlib import Path
import sys


OLD = """        missing = {k: v for k, v in wanted.items() if k not in self._slots}
        if not missing:
            return
        if len(self._slots) + len(missing) > self.max_plans:
            # Every plan a request looks up has to stay resident for the whole
            # denoise loop, so an overflow means the capacity is too small --
            # evicting part of it would only move the failure into lookup().
            self._slots.clear()
            self.plan_lengths.zero_()
        if len(missing) > self.max_plans:
"""

NEW = """        missing = {k: v for k, v in wanted.items() if k not in self._slots}
        if not missing:
            return
        if len(self._slots) + len(missing) > self.max_plans:
            # Every plan a request looks up has to stay resident for the whole
            # denoise loop, so an overflow means the capacity is too small --
            # evicting part of it would only move the failure into lookup().
            self._slots.clear()
            self.plan_lengths.zero_()
            # The old slab may have contained plans shared with this request.
            # Once the slab is cleared, every wanted plan must be rebuilt.
            missing = wanted
        if len(missing) > self.max_plans:
"""


def patch_file(path: Path) -> None:
    source = path.read_text()
    if NEW in source:
        print(f"MiniMax-H3 AdaLN overflow fix already applied: {path}")
        return
    if OLD not in source:
        raise SystemExit(
            "SGLang MiniMax-H3 AdaLN source does not match the validated "
            f"0.5.18 patch context: {path}"
        )
    path.write_text(source.replace(OLD, NEW, 1))
    print(f"Applied MiniMax-H3 AdaLN overflow fix: {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} MINIMAX_H3_SOURCE")
    patch_file(Path(sys.argv[1]).resolve())
