"""Portable adapters used by the WHU-STree competition workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def transfer_instance_labels(
    target_xyz: np.ndarray,
    prediction_xyz: np.ndarray,
    prediction_labels: np.ndarray,
    *,
    tolerance: float = 0.01,
    unassigned_label: int = -1,
) -> np.ndarray:
    """Map labels from a processed cloud back to the original point order.

    TreeLearn may voxelize or filter input points. Codabench instead expects one
    label for every point in the original PLY. A nearest-neighbour lookup is
    therefore used with a strict metric tolerance. Points outside the tolerance
    receive the competition's unassigned sentinel, ``-1``.
    """

    target = np.asarray(target_xyz, dtype=np.float64)
    source = np.asarray(prediction_xyz, dtype=np.float64)
    labels = np.asarray(prediction_labels).reshape(-1)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target_xyz must have shape [N, 3]")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("prediction_xyz must have shape [M, 3]")
    if labels.shape[0] != source.shape[0]:
        raise ValueError("prediction labels must align with prediction_xyz")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if source.shape[0] == 0:
        return np.full(target.shape[0], unassigned_label, dtype=np.int16)

    distances, indices = cKDTree(source).query(target, k=1, workers=1)
    output = np.full(target.shape[0], unassigned_label, dtype=np.int64)
    matched = distances < tolerance
    output[matched] = labels[indices[matched]].astype(np.int64, copy=False)
    if output.size and (
        output.min() < np.iinfo(np.int16).min
        or output.max() > np.iinfo(np.int16).max
    ):
        raise ValueError("instance IDs must fit in int16")
    return output.astype(np.int16, copy=False)


def read_xyz_from_ply(path: str | Path) -> np.ndarray:
    """Read XYZ coordinates from a PLY file without changing point order."""

    try:
        from plyfile import PlyData
    except ImportError as error:  # pragma: no cover - dependency error path
        raise ImportError("Install plyfile with `pip install -e .`") from error
    vertex = PlyData.read(Path(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"{path} does not contain x, y, and z fields")
    return np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(
        np.float64
    )


def read_treelearn_laz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read XYZ and ``treeID`` predictions from a TreeLearn LAS/LAZ file."""

    try:
        import laspy
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Install competition extras with `pip install -e .[competition]`"
        ) from error
    with laspy.open(Path(path)) as stream:
        points = stream.read()
    if "treeID" not in set(points.point_format.extra_dimension_names):
        raise ValueError(f"{path} has no treeID extra dimension")
    xyz = np.column_stack([points.x, points.y, points.z]).astype(np.float64)
    labels = np.asarray(points.treeID, dtype=np.int64)
    return xyz, labels


def convert_ply_to_laz(
    source: str | Path,
    destination: str | Path,
    *,
    labeled: bool,
    reject_abs_coordinate: float | None = 1_000.0,
) -> Path:
    """Convert a WHU-STree PLY into the LAS layout expected by TreeLearn."""

    try:
        import laspy
        from plyfile import PlyData
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Install competition extras with `pip install -e .[competition]`"
        ) from error

    source_path = Path(source)
    vertex = PlyData.read(source_path)["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z"} | ({"tree"} if labeled else set())
    missing = required - names
    if missing:
        raise ValueError(f"{source_path} is missing PLY fields: {sorted(missing)}")

    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(
        np.float64
    )
    keep = np.isfinite(xyz).all(axis=1)
    if reject_abs_coordinate is not None:
        keep &= (np.abs(xyz) < reject_abs_coordinate).all(axis=1)
    xyz = xyz[keep]
    if xyz.shape[0] == 0:
        raise ValueError(f"{source_path} contains no valid coordinates")

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(axis=0)
    output = laspy.LasData(header)
    output.x, output.y, output.z = xyz.T
    if labeled:
        raw_labels = np.asarray(vertex["tree"], dtype=np.int64)[keep]
        unique, inverse = np.unique(raw_labels, return_inverse=True)
        remapped = np.arange(unique.size, dtype=np.uint32)[inverse]
        output.classification = np.where(raw_labels == 0, 2, 4).astype(np.uint8)
        output.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.uint32))
        output.treeID = remapped

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    output.write(destination_path)
    return destination_path
