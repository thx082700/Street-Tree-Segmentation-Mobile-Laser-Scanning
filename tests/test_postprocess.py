import numpy as np

from sparsetree3d.postprocess import PostprocessConfig, cluster_instances


def test_offset_clustering_recovers_two_instances() -> None:
    random = np.random.default_rng(11)
    first = random.normal([0, 0, 2], [0.6, 0.6, 1.0], size=(100, 3))
    second = random.normal([5, 0, 2], [0.6, 0.6, 1.0], size=(100, 3))
    background = random.uniform([-2, -2, 0], [7, 2, 4], size=(20, 3))
    xyz = np.vstack([first, second, background]).astype(np.float32)
    centers = np.vstack(
        [np.tile([0, 0, 2], (100, 1)), np.tile([5, 0, 2], (100, 1)), background]
    )
    offsets = centers - xyz
    probability = np.concatenate([np.full(200, 0.95), np.full(20, 0.05)])
    labels = cluster_instances(
        xyz,
        probability,
        offsets,
        PostprocessConfig(
            cluster_eps=0.25,
            cluster_min_samples=3,
            min_cluster_points=10,
            max_horizontal_extent=5.0,
        ),
    )
    assert set(np.unique(labels)) == {0, 1, 2}
    assert np.unique(labels[:100]).size == 1
    assert np.unique(labels[100:200]).size == 1
    assert np.all(labels[200:] == 0)


def test_no_tree_points_returns_background() -> None:
    xyz = np.zeros((5, 3), dtype=np.float32)
    labels = cluster_instances(xyz, np.zeros(5), np.zeros_like(xyz))
    assert np.array_equal(labels, np.zeros(5, dtype=np.int64))

