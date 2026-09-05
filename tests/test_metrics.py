import numpy as np
import pytest

from sparsetree3d.metrics import aggregate_metrics, evaluate_instances


def test_perfect_instances() -> None:
    ground_truth = np.array([0, 1, 1, 2, 2, 2])
    prediction = np.array([0, 8, 8, 3, 3, 3])
    result = evaluate_instances(ground_truth, prediction, iou_threshold=0.75)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.coverage == 1.0
    assert result.weighted_coverage == 1.0


def test_false_positive_and_missed_tree() -> None:
    ground_truth = np.array([0, 1, 1, 2, 2, 0, 0])
    prediction = np.array([0, 4, 4, 0, 0, 7, 7])
    result = evaluate_instances(ground_truth, prediction, iou_threshold=0.75)
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)


def test_aggregate_weights_coverage_by_instances() -> None:
    first = evaluate_instances(np.array([1, 1]), np.array([1, 1]))
    second = evaluate_instances(np.array([1, 2]), np.array([0, 0]))
    result = aggregate_metrics([first, second])
    assert result.ground_truth_instances == 3
    assert result.coverage == pytest.approx(1 / 3)

