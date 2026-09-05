"""Installed command-line entry points for evaluation and submission packaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sparsetree3d.io import make_submission_archive
from sparsetree3d.metrics import aggregate_metrics, evaluate_instances


def evaluate_directories(
    ground_truth_dir: Path, prediction_dir: Path, *, iou_threshold: float
):
    results = []
    for ground_truth_path in sorted(ground_truth_dir.glob("*.npy")):
        prediction_path = prediction_dir / ground_truth_path.name
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing prediction: {prediction_path}")
        ground_truth = np.load(ground_truth_path, allow_pickle=False)
        prediction = np.load(prediction_path, allow_pickle=False)
        result = evaluate_instances(ground_truth, prediction, iou_threshold=iou_threshold)
        results.append(result)
        print(json.dumps({"scene": ground_truth_path.stem, **result.to_dict()}, sort_keys=True))
    aggregate = aggregate_metrics(results)
    print(json.dumps({"scene": "ALL", **aggregate.to_dict()}, sort_keys=True))
    return aggregate


def evaluate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate WHU-STree instance predictions")
    parser.add_argument("ground_truth_dir", type=Path)
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.75)
    args = parser.parse_args(argv)
    evaluate_directories(
        args.ground_truth_dir, args.prediction_dir, iou_threshold=args.iou_threshold
    )


def submission_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a validated Codabench ZIP")
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    archive = make_submission_archive(args.prediction_dir.glob("*.npy"), args.archive)
    print(archive)

