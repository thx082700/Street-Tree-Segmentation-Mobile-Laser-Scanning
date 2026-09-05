"""WHU-STree point-cloud and competition-output I/O."""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PointCloud:
    """A point cloud and the labels available in a WHU-STree PLY file."""

    xyz: np.ndarray
    intensity: np.ndarray
    instance: np.ndarray | None = None
    species: np.ndarray | None = None


def read_ply(path: str | Path, *, require_instance: bool = False) -> PointCloud:
    """Read WHU-STree PLY attributes without loading any image data.

    The current dataset release documents the fields
    ``[x, y, z, intensity, tree, label]``. ``tree`` and ``label`` remain optional
    here so the same reader can process unlabeled competition test clouds.
    """

    try:
        from plyfile import PlyData
    except ImportError as error:  # pragma: no cover - installation error path
        raise ImportError("Install the core package with `pip install -e .`") from error

    source = Path(path)
    vertex = PlyData.read(source)["vertex"].data
    names = set(vertex.dtype.names or ())
    missing = {"x", "y", "z"} - names
    if missing:
        raise ValueError(f"{source} is missing PLY fields: {sorted(missing)}")

    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    intensity = (
        np.asarray(vertex["intensity"], dtype=np.float32)
        if "intensity" in names
        else np.zeros(xyz.shape[0], dtype=np.float32)
    )
    instance = (
        np.asarray(vertex["tree"], dtype=np.int64) if "tree" in names else None
    )
    species = (
        np.asarray(vertex["label"], dtype=np.int64) if "label" in names else None
    )
    if require_instance and instance is None:
        raise ValueError(f"{source} has no 'tree' instance-label field")
    return PointCloud(xyz=xyz, intensity=intensity, instance=instance, species=species)


def build_features(xyz: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Build four robust per-point features used by the sparse backbone.

    Coordinates still define the sparse tensor. The feature vector contains a
    constant occupancy channel, robustly normalized intensity, relative height,
    and horizontal radius. Normalizers are computed per scene.
    """

    xyz = np.asarray(xyz, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or intensity.shape[0] != xyz.shape[0]:
        raise ValueError("xyz must be [N, 3] and intensity must contain N values")
    if xyz.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    centered = xyz - np.median(xyz, axis=0, keepdims=True)
    xy_radius = np.linalg.norm(centered[:, :2], axis=1)
    height = xyz[:, 2] - np.percentile(xyz[:, 2], 1)

    def robust_scale(values: np.ndarray) -> np.ndarray:
        median = np.median(values)
        scale = np.percentile(np.abs(values - median), 90)
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        return np.clip((values - median) / scale, -3.0, 3.0)

    features = np.column_stack(
        [
            np.ones(xyz.shape[0], dtype=np.float32),
            robust_scale(intensity),
            robust_scale(height),
            robust_scale(xy_radius),
        ]
    )
    return features.astype(np.float32, copy=False)


def competition_stem(path: str | Path) -> str:
    """Convert ``<road>/PCD/<cloud>.ply`` into ``<road>_<cloud>``."""

    source = Path(path)
    road = source.parent.parent.name if source.parent.name.lower() == "pcd" else ""
    if not road:
        match = re.fullmatch(r"(.+)_([^_]+)", source.stem)
        if match:
            return source.stem
        raise ValueError(
            f"Cannot infer road ID from {source}; expected <road>/PCD/<cloud>.ply"
        )
    return f"{road}_{source.stem}"


def save_prediction(path: str | Path, labels: np.ndarray) -> Path:
    """Save a competition-format int16 instance vector."""

    destination = Path(path)
    if destination.suffix != ".npy":
        destination = destination.with_suffix(".npy")
    values = np.asarray(labels)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("Prediction labels must have shape [N] or [N, 1]")
    if values.size and (values.min() < 0 or values.max() > np.iinfo(np.int16).max):
        raise ValueError("Instance IDs must fit in non-negative int16")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, values.astype(np.int16, copy=False))
    return destination


def validate_prediction_file(path: str | Path) -> tuple[int, int]:
    """Validate a saved prediction and return ``(number_of_points, instances)``."""

    source = Path(path)
    if not re.fullmatch(r"[^_]+_[^_]+\.npy", source.name):
        raise ValueError(f"Invalid competition filename: {source.name}")
    labels = np.load(source, allow_pickle=False)
    if labels.dtype != np.int16:
        raise ValueError(f"{source.name} must use int16, got {labels.dtype}")
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise ValueError(f"{source.name} must have shape [N] or [N, 1]")
    if labels.size and labels.min() < 0:
        raise ValueError(f"{source.name} contains negative instance IDs")
    return int(labels.size), int(np.unique(labels[labels > 0]).size)


def make_submission_archive(
    prediction_paths: Iterable[str | Path], destination: str | Path
) -> Path:
    """Validate predictions and write a flat ZIP archive for Codabench."""

    files = sorted(Path(path) for path in prediction_paths)
    if not files:
        raise ValueError("No .npy predictions were provided")
    names = [path.name for path in files]
    if len(names) != len(set(names)):
        raise ValueError("Prediction filenames must be unique in the flat archive")
    for path in files:
        validate_prediction_file(path)

    archive = Path(destination)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, arcname=path.name)
    return archive
