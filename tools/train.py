#!/usr/bin/env python3
"""Train SparseTree3D."""

from __future__ import annotations

import argparse

from sparsetree3d.config import load_config, require_section
from sparsetree3d.data import WHUSTreeDataset, read_manifest
from sparsetree3d.engine import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/whu_stree.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data = require_section(config, "data")
    root = data["root"]
    voxel_size = float(data["voxel_size"])
    train_dataset = WHUSTreeDataset(
        read_manifest(root, data["train_manifest"]), voxel_size, labeled=True
    )
    validation_dataset = WHUSTreeDataset(
        read_manifest(root, data["val_manifest"]), voxel_size, labeled=True
    )
    print(train_model(train_dataset, validation_dataset, config))


if __name__ == "__main__":
    main()

