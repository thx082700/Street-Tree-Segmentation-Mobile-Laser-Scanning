"""Training and inference loops for the sparse-convolution model."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from sparsetree3d.config import require_section
from sparsetree3d.data import VoxelSample
from sparsetree3d.io import competition_stem, save_prediction
from sparsetree3d.losses import semantic_offset_loss
from sparsetree3d.model import build_model, load_compatible_weights
from sparsetree3d.postprocess import PostprocessConfig, cluster_instances


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _training_dependencies():
    try:
        import MinkowskiEngine as ME
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
    except ImportError as error:  # pragma: no cover - depends on CUDA build
        raise ImportError(
            "Training and neural inference require PyTorch, MinkowskiEngine, and tqdm"
        ) from error
    return torch, ME, DataLoader, tqdm


def resolve_device(requested: str):
    torch, _, _, _ = _training_dependencies()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _sparse_batch(samples: list[VoxelSample], device: Any):
    torch, ME, _, _ = _training_dependencies()
    coordinates = ME.utils.batched_coordinates([sample.coords for sample in samples])
    features = torch.from_numpy(np.concatenate([sample.features for sample in samples])).to(device)
    sparse = ME.SparseTensor(features=features, coordinates=coordinates, device=device)
    semantic = None
    offsets = None
    if all(sample.semantic is not None and sample.offsets is not None for sample in samples):
        semantic = torch.from_numpy(
            np.concatenate([sample.semantic for sample in samples])  # type: ignore[arg-type]
        ).long().to(device)
        offsets = torch.from_numpy(
            np.concatenate([sample.offsets for sample in samples])  # type: ignore[arg-type]
        ).float().to(device)
    return sparse, semantic, offsets


def _run_epoch(model, loader, optimizer, config: dict[str, Any], device, *, training: bool):
    torch, _, _, tqdm = _training_dependencies()
    loss_config = require_section(config, "loss")
    model.train(training)
    totals = {"loss": 0.0, "semantic": 0.0, "offset": 0.0, "batches": 0}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for samples in tqdm(loader, leave=False):
            sparse, semantic_target, offset_target = _sparse_batch(samples, device)
            if semantic_target is None or offset_target is None:
                raise ValueError("Training and validation samples require instance labels")
            if training:
                optimizer.zero_grad(set_to_none=True)
            semantic_logits, predicted_offsets = model(sparse)
            loss, parts = semantic_offset_loss(
                semantic_logits,
                predicted_offsets,
                semantic_target,
                offset_target,
                class_weights=loss_config["semantic_class_weights"],
                offset_weight=float(loss_config.get("offset_weight", 1.0)),
            )
            if training:
                loss.backward()
                optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["semantic"] += float(parts["semantic"])
            totals["offset"] += float(parts["offset"])
            totals["batches"] += 1
    denominator = max(int(totals.pop("batches")), 1)
    return {name: value / denominator for name, value in totals.items()}


def train_model(train_dataset, validation_dataset, config: dict[str, Any]) -> Path:
    """Train the joint model and save the best validation-loss checkpoint."""

    torch, _, DataLoader, _ = _training_dependencies()
    model_config = require_section(config, "model")
    train_config = require_section(config, "train")
    seed_everything(int(config.get("seed", 2025)))
    device = resolve_device(str(train_config.get("device", "cuda")))
    model = build_model(
        in_channels=int(model_config.get("in_channels", 4)),
        base_channels=int(model_config.get("base_channels", 32)),
    ).to(device)
    pretrained = model_config.get("pretrained_backbone")
    if pretrained:
        load_compatible_weights(model, pretrained)

    loader_options = {
        "batch_size": int(train_config.get("batch_size", 2)),
        "num_workers": int(require_section(config, "data").get("num_workers", 4)),
        "collate_fn": lambda batch: batch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_config.get("epochs", 80))
    )
    checkpoint_dir = Path(train_config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pth"
    best_loss = float("inf")

    for epoch in range(1, int(train_config.get("epochs", 80)) + 1):
        training_metrics = _run_epoch(model, train_loader, optimizer, config, device, training=True)
        validation_metrics = _run_epoch(
            model, validation_loader, optimizer, config, device, training=False
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train": training_metrics,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(record, sort_keys=True))
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "config": config}, best_path
            )
    return best_path


def _postprocess_config(config: dict[str, Any]) -> PostprocessConfig:
    values = require_section(config, "postprocess")
    keys = PostprocessConfig.__dataclass_fields__.keys()
    return PostprocessConfig(**{key: values[key] for key in keys if key in values})


def infer_dataset(dataset, checkpoint: str | Path, output_dir: str | Path, config):
    """Run voxel-level inference and save point-level competition predictions."""

    torch, _, _, tqdm = _training_dependencies()
    model_config = require_section(config, "model")
    train_config = require_section(config, "train")
    device = resolve_device(str(train_config.get("device", "cuda")))
    model = build_model(
        in_channels=int(model_config.get("in_channels", 4)),
        base_channels=int(model_config.get("base_channels", 32)),
    ).to(device)
    load_compatible_weights(model, checkpoint)
    model.eval()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    postprocess_config = _postprocess_config(config)

    with torch.no_grad():
        for index in tqdm(range(len(dataset))):
            sample = dataset[index]
            sparse, _, _ = _sparse_batch([sample], device)
            semantic_logits, offsets = model(sparse)
            probabilities = torch.softmax(semantic_logits, dim=1)[:, 1].cpu().numpy()
            voxel_labels = cluster_instances(
                sample.xyz, probabilities, offsets.cpu().numpy(), postprocess_config
            )
            point_labels = voxel_labels[sample.inverse]
            output_path = destination / f"{competition_stem(sample.path)}.npy"
            written.append(save_prediction(output_path, point_labels))
    return written

