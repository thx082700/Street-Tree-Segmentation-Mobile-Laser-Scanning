#!/usr/bin/env python3
"""Run the eight scene-specific competition inference configurations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_SCENES = (
    "05_3-2",
    "06_2-1",
    "09_1-1",
    "12_2",
    "13_2",
    "15_1",
    "15_2-1",
    "17_3-3",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    args = parser.parse_args()

    for scene in args.scenes:
        config = Path("configs/competition/pipeline") / f"pipeline_{scene}.yaml"
        if not config.exists():
            raise FileNotFoundError(config)
        command = [
            sys.executable,
            "tools/competition/pipeline/pipeline.py",
            "--config",
            str(config),
        ]
        print("running", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
