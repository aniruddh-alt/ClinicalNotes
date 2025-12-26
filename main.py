"""
Dev-friendly entrypoint.

In normal use, run the installed console script:
  - `uv run clinicalnotes ...`

This file supports running `python main.py ...` without installing the package by
adding `src/` to `sys.path`.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = (Path(__file__).resolve().parent / "src").as_posix()
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from clinicalnotes_1.cli import main  # type: ignore[import-not-found]  # noqa: E402


if __name__ == "__main__":
    main()
