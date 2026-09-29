import numpy as np
import pytest

from sparsetree3d.adapters import transfer_instance_labels


def test_transfer_instance_labels_preserves_original_order() -> None:
    prediction_xyz = np.array([[2, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
    prediction_labels = np.array([20, 10, 10])
    target_xyz = np.array([[0, 0, 0], [1.004, 0, 0], [9, 0, 0]], dtype=float)
    result = transfer_instance_labels(
        target_xyz, prediction_xyz, prediction_labels, tolerance=0.01
    )
    assert result.dtype == np.int16
    assert result.tolist() == [10, 10, -1]


def test_transfer_instance_labels_validates_shapes() -> None:
    with pytest.raises(ValueError):
        transfer_instance_labels(np.zeros((2, 2)), np.zeros((2, 3)), np.zeros(2))
