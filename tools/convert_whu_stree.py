#!/usr/bin/env python3
"""Convert a WHU-STree train or test tree into TreeLearn's LAS/LAZ layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from sparsetree3d.adapters import convert_ply_to_laz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Directory containing <road>/PCD/*.ply")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--labeled", action="store_true", help="Read the PLY tree field")
    parser.add_argument(
        "--keep-large-coordinates",
        action="store_true",
        help="Do not reject the four known competition files with extreme coordinates",
    )
    args = parser.parse_args()

    files = sorted(args.dataset_root.glob("*/PCD/*.ply"))
    if not files:
        raise SystemExit(f"No <road>/PCD/*.ply files found under {args.dataset_root}")
    for source in files:
        road = source.parent.parent.name
        stem = f"{road}_{source.stem}"
        if args.labeled:
            destination = args.output_root / "forests" / f"{stem}.laz"
        else:
            destination = args.output_root / f"pipeline_{stem}" / "forest" / f"{stem}.laz"
        convert_ply_to_laz(
            source,
            destination,
            labeled=args.labeled,
            reject_abs_coordinate=None if args.keep_large_coordinates else 1_000.0,
        )
        print(destination)


if __name__ == "__main__":
    main()
