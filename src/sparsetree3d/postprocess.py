"""Offset-guided clustering and size-aware instance refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans


@dataclass(frozen=True)
class PostprocessConfig:
    tree_probability_threshold: float = 0.55
    cluster_eps: float = 0.65
    cluster_min_samples: int = 8
    min_cluster_points: int = 35
    fragment_merge_radius: float = 2.0
    max_horizontal_extent: float = 10.0
    max_cluster_points: int = 12_000
    max_splits: int = 4


def _validate_inputs(
    xyz: np.ndarray, tree_probability: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float32)
    probability = np.asarray(tree_probability, dtype=np.float32).reshape(-1)
    offsets = np.asarray(offsets, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if offsets.shape != xyz.shape or probability.shape[0] != xyz.shape[0]:
        raise ValueError("tree_probability and offsets must align with xyz")
    return xyz, probability, offsets


def _split_oversized(
    xyz: np.ndarray, labels: np.ndarray, config: PostprocessConfig
) -> np.ndarray:
    """Split implausibly large clusters in the horizontal plane."""

    refined = labels.copy()
    next_label = int(refined.max(initial=-1)) + 1
    for label in np.unique(labels[labels >= 0]):
        indices = np.flatnonzero(labels == label)
        horizontal = xyz[indices, :2]
        extent = float(np.linalg.norm(np.ptp(horizontal, axis=0)))
        point_splits = int(np.ceil(indices.size / max(config.max_cluster_points, 1)))
        extent_splits = int(np.ceil(extent / max(config.max_horizontal_extent, 1e-6)))
        splits = min(max(point_splits, extent_splits, 1), config.max_splits)
        if splits <= 1 or indices.size < splits * config.cluster_min_samples:
            continue
        children = KMeans(n_clusters=splits, n_init=10, random_state=2025).fit_predict(horizontal)
        refined[indices] = next_label + children
        next_label += splits
    return refined


def _merge_fragments(
    xyz: np.ndarray, labels: np.ndarray, config: PostprocessConfig
) -> np.ndarray:
    """Merge small fragments and DBSCAN noise into the nearest stable tree."""

    refined = labels.copy()
    cluster_ids, counts = np.unique(refined[refined >= 0], return_counts=True)
    stable_ids = cluster_ids[counts >= config.min_cluster_points]
    if stable_ids.size == 0:
        return np.full(labels.shape, -1, dtype=np.int64)

    centroids = np.stack([xyz[refined == label].mean(axis=0) for label in stable_ids])
    fragment_ids = cluster_ids[counts < config.min_cluster_points]
    for label in fragment_ids:
        indices = np.flatnonzero(refined == label)
        center = xyz[indices].mean(axis=0)
        nearest = int(np.argmin(np.linalg.norm(centroids - center, axis=1)))
        distance = float(np.linalg.norm(centroids[nearest] - center))
        refined[indices] = stable_ids[nearest] if distance <= config.fragment_merge_radius else -1

    noise = np.flatnonzero(refined < 0)
    if noise.size:
        distances = np.linalg.norm(xyz[noise, None, :] - centroids[None, :, :], axis=2)
        nearest = distances.argmin(axis=1)
        accepted = distances[np.arange(noise.size), nearest] <= config.fragment_merge_radius
        refined[noise[accepted]] = stable_ids[nearest[accepted]]
    return refined


def _renumber(labels: np.ndarray) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=np.int64)
    for new_id, old_id in enumerate(np.unique(labels[labels >= 0]), start=1):
        output[labels == old_id] = new_id
    return output


def cluster_instances(
    xyz: np.ndarray,
    tree_probability: np.ndarray,
    offsets: np.ndarray,
    config: PostprocessConfig | None = None,
) -> np.ndarray:
    """Convert semantic and centroid-offset predictions into instance IDs.

    Tree voxels first move toward their predicted centroids. DBSCAN groups the
    shifted points. The refinement stage then splits clusters with implausible
    horizontal size and merges small fragments, addressing the failure modes
    highlighted in the competition report.
    """

    config = config or PostprocessConfig()
    xyz, probability, offsets = _validate_inputs(xyz, tree_probability, offsets)
    output = np.zeros(xyz.shape[0], dtype=np.int64)
    tree_indices = np.flatnonzero(probability >= config.tree_probability_threshold)
    if tree_indices.size == 0:
        return output

    shifted = xyz[tree_indices] + offsets[tree_indices]
    labels = DBSCAN(
        eps=config.cluster_eps,
        min_samples=config.cluster_min_samples,
        algorithm="kd_tree",
        n_jobs=1,
    ).fit_predict(shifted)
    labels = _split_oversized(xyz[tree_indices], labels, config)
    labels = _merge_fragments(xyz[tree_indices], labels, config)
    output[tree_indices] = _renumber(labels)
    return output
