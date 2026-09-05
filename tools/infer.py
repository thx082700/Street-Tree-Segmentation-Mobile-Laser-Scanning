#!/usr/bin/env python3
"""Run SparseTree3D and write competition-format instance IDs."""

from __future__ import annotations

import argparse

from sparsetree3d.config import load_config, require_section
from sparsetree3d.data import WHUSTreeDataset, read_manifest
from sparsetree3d.engine import infer_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/whu_stree.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data = require_section(config, "data")
    manifest = args.manifest or data["test_manifest"]
    dataset = WHUSTreeDataset(
        read_manifest(data["root"], manifest), float(data["voxel_size"]), labeled=False
    )
    written = infer_dataset(dataset, args.checkpoint, args.output_dir, config)
    print(f"wrote {len(written)} predictions to {args.output_dir}")


if __name__ == "__main__":
    main()

