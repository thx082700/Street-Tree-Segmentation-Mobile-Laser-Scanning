"""Voxelization, manifests, and PyTorch dataset adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sparsetree3d.io import build_features, read_ply


@dataclass(frozen=True)
class VoxelSample:
    coords: np.ndarray
    features: np.ndarray
    xyz: np.ndarray
    inverse: np.ndarray
    semantic: np.ndarray | None
    offsets: np.ndarray | None
    instance: np.ndarray | None
    path: Path


def centroid_offsets(xyz: np.ndarray, instance: np.ndarray) -> np.ndarray:
    """Compute the vector from every tree point to its instance centroid."""

    xyz = np.asarray(xyz, dtype=np.float32)
    instance = np.asarray(instance, dtype=np.int64).reshape(-1)
    if xyz.shape != (instance.size, 3):
        raise ValueError("xyz and instance arrays have incompatible shapes")
    offsets = np.zeros_like(xyz, dtype=np.float32)
    for instance_id in np.unique(instance[instance > 0]):
        mask = instance == instance_id
        offsets[mask] = xyz[mask].mean(axis=0, keepdims=True) - xyz[mask]
    return offsets


def voxelize_cloud(path: str | Path, voxel_size: float, *, labeled: bool) -> VoxelSample:
    """Quantize one PLY file while preserving an inverse map to original points."""

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    cloud = read_ply(path, require_instance=labeled)
    origin = cloud.xyz.min(axis=0, keepdims=True) if cloud.xyz.size else np.zeros((1, 3))
    quantized = np.floor((cloud.xyz - origin) / voxel_size).astype(np.int32)
    _, unique_indices, inverse = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True
    )
    features = build_features(cloud.xyz, cloud.intensity)

    semantic = None
    offsets = None
    if cloud.instance is not None:
        semantic_all = (cloud.instance > 0).astype(np.int64)
        offsets_all = centroid_offsets(cloud.xyz, cloud.instance)
        semantic = semantic_all[unique_indices]
        offsets = offsets_all[unique_indices]

    return VoxelSample(
        coords=quantized[unique_indices],
        features=features[unique_indices],
        xyz=cloud.xyz[unique_indices],
        inverse=inverse.astype(np.int64, copy=False),
        semantic=semantic,
        offsets=offsets,
        instance=cloud.instance,
        path=Path(path),
    )


def read_manifest(root: str | Path, manifest: str | Path) -> list[Path]:
    """Resolve a newline-delimited manifest relative to the dataset root."""

    root_path = Path(root)
    manifest_path = Path(manifest)
    entries: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.split("#", 1)[0].strip()
            if line:
                entries.append(root_path / line)
    return entries


def discover_ply_files(root: str | Path) -> list[Path]:
    """Discover the official ``<road>/PCD/*.ply`` layout."""

    return sorted(Path(root).glob("*/PCD/*.ply"))


def deterministic_split(
    paths: Iterable[Path], *, validation_fraction: float = 0.2, seed: int = 2025
) -> tuple[list[Path], list[Path]]:
    """Split by road ID so trajectories from one road do not leak across sets."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    groups: dict[str, list[Path]] = {}
    for path in paths:
        road = path.parent.parent.name
        groups.setdefault(road, []).append(path)

    train: list[Path] = []
    validation: list[Path] = []
    for road, members in sorted(groups.items()):
        digest = hashlib.sha256(f"{seed}:{road}".encode()).digest()
        score = int.from_bytes(digest[:8], "big") / 2**64
        target = validation if score < validation_fraction else train
        target.extend(sorted(members))
    return train, validation


class WHUSTreeDataset:
    """A minimal indexable dataset; importable even when PyTorch is absent."""

    def __init__(self, paths: Iterable[Path], voxel_size: float, *, labeled: bool = True):
        self.paths = list(paths)
        self.voxel_size = float(voxel_size)
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> VoxelSample:
        return voxelize_cloud(self.paths[index], self.voxel_size, labeled=self.labeled)
