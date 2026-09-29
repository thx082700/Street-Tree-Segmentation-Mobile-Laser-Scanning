#!/usr/bin/env python3
"""Back-project TreeLearn results and create a Codabench submission archive."""

from __future__ import annotations

import argparse
from pathlib import Path

from sparsetree3d.adapters import (
    read_treelearn_laz,
    read_xyz_from_ply,
    transfer_instance_labels,
)
from sparsetree3d.io import make_submission_archive, save_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_root", type=Path, help="WHU-STree test root with <road>/PCD/*.ply")
    parser.add_argument("pipeline_root", type=Path, help="Root containing pipeline_* results")
    parser.add_argument("output_zip", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    prediction_dir = args.output_zip.parent / f"{args.output_zip.stem}_files"
    written: list[Path] = []
    for source in sorted(args.test_root.glob("*/PCD/*.ply")):
        road = source.parent.parent.name
        competition_stem = f"{road}_{source.stem}"
        prediction_path = (
            args.pipeline_root
            / f"pipeline_{competition_stem}"
            / "results"
            / "full_forest"
            / f"{competition_stem}.laz"
        )
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing TreeLearn result: {prediction_path}")
        target_xyz = read_xyz_from_ply(source)
        prediction_xyz, prediction_labels = read_treelearn_laz(prediction_path)
        labels = transfer_instance_labels(
            target_xyz,
            prediction_xyz,
            prediction_labels,
            tolerance=args.tolerance,
        )
        written.append(save_prediction(prediction_dir / competition_stem, labels))

    archive = make_submission_archive(written, args.output_zip)
    print(f"wrote {len(written)} scenes to {archive}")


if __name__ == "__main__":
    main()
