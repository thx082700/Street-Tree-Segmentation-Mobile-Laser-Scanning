#!/usr/bin/env python3
"""Create road-disjoint train/validation manifests from a WHU-STree release."""

from __future__ import annotations

import argparse
from pathlib import Path

from sparsetree3d.data import deterministic_split, discover_ply_files


def write_manifest(path: Path, root: Path, entries: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{entry.relative_to(root)}\n" for entry in entries), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    paths = discover_ply_files(args.root)
    if not paths:
        raise FileNotFoundError(f"No <road>/PCD/*.ply files found under {args.root}")
    train, validation = deterministic_split(
        paths, validation_fraction=args.validation_fraction, seed=args.seed
    )
    write_manifest(args.output_dir / "train.txt", args.root, train)
    write_manifest(args.output_dir / "val.txt", args.root, validation)
    print(f"train={len(train)} validation={len(validation)}")


if __name__ == "__main__":
    main()

