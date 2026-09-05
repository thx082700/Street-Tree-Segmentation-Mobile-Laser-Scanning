"""Instance-level and point-level metrics used by WHU-STree Track 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class InstanceMetrics:
    """Competition metrics plus counts needed for dataset-level aggregation."""

    precision: float
    recall: float
    f1: float
    coverage: float
    weighted_coverage: float
    true_positives: int
    false_positives: int
    false_negatives: int
    ground_truth_instances: int
    predicted_instances: int
    ground_truth_points: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _positive_instance_ids(labels: np.ndarray) -> np.ndarray:
    return np.unique(labels[labels > 0])


def instance_iou_matrix(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the pairwise IoU matrix and per-instance point counts."""

    gt = np.asarray(ground_truth).reshape(-1)
    pred = np.asarray(prediction).reshape(-1)
    if gt.shape != pred.shape:
        raise ValueError("Ground-truth and prediction arrays must have equal length")

    valid = (gt >= 0) & (pred >= 0)
    gt = gt[valid]
    pred = pred[valid]
    gt_ids = _positive_instance_ids(gt)
    pred_ids = _positive_instance_ids(pred)
    gt_counts = np.array([(gt == value).sum() for value in gt_ids], dtype=np.int64)
    pred_counts = np.array([(pred == value).sum() for value in pred_ids], dtype=np.int64)
    intersections = np.zeros((gt_ids.size, pred_ids.size), dtype=np.int64)

    if gt_ids.size and pred_ids.size:
        overlap = (gt > 0) & (pred > 0)
        gt_index = np.searchsorted(gt_ids, gt[overlap])
        pred_index = np.searchsorted(pred_ids, pred[overlap])
        np.add.at(intersections, (gt_index, pred_index), 1)

    unions = gt_counts[:, None] + pred_counts[None, :] - intersections
    iou = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )
    return iou, gt_ids, pred_ids, gt_counts


def evaluate_instances(
    ground_truth: np.ndarray, prediction: np.ndarray, *, iou_threshold: float = 0.75
) -> InstanceMetrics:
    """Evaluate one scene using one-to-one IoU matching.

    Positive IDs denote tree instances, ``0`` denotes background, and negative
    labels are ignored. Hungarian matching prevents one prediction from matching
    multiple ground-truth trees. Coverage metrics use each ground-truth tree's
    best IoU, independent of the detection threshold.
    """

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    gt = np.asarray(ground_truth).reshape(-1)
    pred = np.asarray(prediction).reshape(-1)
    iou, gt_ids, pred_ids, gt_counts = instance_iou_matrix(gt, pred)

    true_positives = 0
    if iou.size:
        rows, cols = linear_sum_assignment(1.0 - iou)
        true_positives = int((iou[rows, cols] >= iou_threshold).sum())

    false_positives = int(pred_ids.size - true_positives)
    false_negatives = int(gt_ids.size - true_positives)
    precision = true_positives / pred_ids.size if pred_ids.size else 0.0
    recall = true_positives / gt_ids.size if gt_ids.size else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    best_iou = iou.max(axis=1) if pred_ids.size else np.zeros(gt_ids.size)
    coverage = float(best_iou.mean()) if gt_ids.size else 0.0
    gt_tree_points = int(gt_counts.sum())
    weighted_coverage = (
        float(np.dot(best_iou, gt_counts) / gt_tree_points) if gt_tree_points else 0.0
    )
    return InstanceMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        coverage=coverage,
        weighted_coverage=weighted_coverage,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        ground_truth_instances=int(gt_ids.size),
        predicted_instances=int(pred_ids.size),
        ground_truth_points=gt_tree_points,
    )


def aggregate_metrics(results: list[InstanceMetrics]) -> InstanceMetrics:
    """Aggregate scene metrics without giving small scenes disproportionate weight."""

    if not results:
        raise ValueError("At least one scene result is required")
    true_positives = sum(item.true_positives for item in results)
    false_positives = sum(item.false_positives for item in results)
    false_negatives = sum(item.false_negatives for item in results)
    gt_instances = sum(item.ground_truth_instances for item in results)
    pred_instances = sum(item.predicted_instances for item in results)
    gt_points = sum(item.ground_truth_points for item in results)

    precision = true_positives / pred_instances if pred_instances else 0.0
    recall = true_positives / gt_instances if gt_instances else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    coverage = (
        sum(item.coverage * item.ground_truth_instances for item in results) / gt_instances
        if gt_instances
        else 0.0
    )
    weighted_coverage = (
        sum(item.weighted_coverage * item.ground_truth_points for item in results) / gt_points
        if gt_points
        else 0.0
    )
    return InstanceMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        coverage=float(coverage),
        weighted_coverage=float(weighted_coverage),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        ground_truth_instances=gt_instances,
        predicted_instances=pred_instances,
        ground_truth_points=gt_points,
    )

