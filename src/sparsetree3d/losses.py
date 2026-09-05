"""Joint semantic and centroid-offset objective from the competition report."""

from __future__ import annotations

from typing import Any


def semantic_offset_loss(
    semantic_logits: Any,
    predicted_offsets: Any,
    semantic_target: Any,
    offset_target: Any,
    *,
    class_weights: list[float] | tuple[float, ...],
    offset_weight: float = 1.0,
):
    """Compute weighted cross entropy plus tree-point offset MSE."""

    import torch
    import torch.nn.functional as functional

    weights = torch.as_tensor(
        class_weights, dtype=semantic_logits.dtype, device=semantic_logits.device
    )
    semantic_loss = functional.cross_entropy(semantic_logits, semantic_target, weight=weights)
    tree_mask = semantic_target == 1
    if tree_mask.any():
        offset_loss = functional.mse_loss(predicted_offsets[tree_mask], offset_target[tree_mask])
    else:
        offset_loss = predicted_offsets.sum() * 0.0
    total = semantic_loss + float(offset_weight) * offset_loss
    return total, {"semantic": semantic_loss.detach(), "offset": offset_loss.detach()}
