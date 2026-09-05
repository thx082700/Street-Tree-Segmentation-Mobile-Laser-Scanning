#!/usr/bin/env python3
"""CPU-only smoke test for the offset-guided clustering stage."""

from __future__ import annotations

import numpy as np

from sparsetree3d.metrics import evaluate_instances
from sparsetree3d.postprocess import PostprocessConfig, cluster_instances


def synthetic_scene(seed: int = 7):
    random = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0, 3.5], [5.0, 0.4, 4.0], [10.0, -0.3, 3.8]])
    clouds = []
    labels = []
    offsets = []
    for instance_id, center in enumerate(centers, start=1):
        points = random.normal(size=(250, 3)) * np.array([1.0, 0.8, 2.0]) + center
        clouds.append(points)
        labels.append(np.full(points.shape[0], instance_id))
        offsets.append(center - points + random.normal(scale=0.08, size=points.shape))

    background = random.uniform([-2, -4, 0], [12, 4, 8], size=(150, 3))
    xyz = np.vstack([*clouds, background]).astype(np.float32)
    ground_truth = np.concatenate([*labels, np.zeros(background.shape[0], dtype=np.int64)])
    predicted_offsets = np.vstack([*offsets, np.zeros_like(background)]).astype(np.float32)
    probabilities = np.concatenate(
        [np.full(sum(cloud.shape[0] for cloud in clouds), 0.95), np.full(background.shape[0], 0.05)]
    ).astype(np.float32)
    return xyz, probabilities, predicted_offsets, ground_truth


def main() -> None:
    xyz, probabilities, offsets, ground_truth = synthetic_scene()
    prediction = cluster_instances(
        xyz,
        probabilities,
        offsets,
        PostprocessConfig(
            cluster_eps=0.5,
            cluster_min_samples=5,
            min_cluster_points=20,
            max_horizontal_extent=20.0,
        ),
    )
    result = evaluate_instances(ground_truth, prediction, iou_threshold=0.75)
    print(result.to_dict())


if __name__ == "__main__":
    main()
