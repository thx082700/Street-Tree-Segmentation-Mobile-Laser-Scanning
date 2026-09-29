#!/usr/bin/env python3
"""Run the competition TreeLearn inference entry point."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    entrypoint = Path(__file__).parent / "competition" / "pipeline" / "pipeline.py"
    runpy.run_path(str(entrypoint), run_name="__main__")
