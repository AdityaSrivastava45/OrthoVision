"""Loss functions for multi-label classification."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("orthovision.training.losses")

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy with logits and optional positive class weighting.

    Args:
        logits: [B, 12] - model outputs (before sigmoid)
        targets: [B, 12] - binary labels (0 or 1), NaN for missing
        pos_weight: [12] - weight for positive class per label

    Returns:
        Scalar loss
    """
    # Handle NaN targets (missing labels) - ignore them in loss
    valid_mask = ~torch.isnan(targets)

    if pos_weight is not None:
        # BCEWithLogitsLoss with per-class pos_weight
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.nan_to_num(targets, nan=0.0),
            pos_weight=pos_weight,
            reduction='none'
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.nan_to_num(targets, nan=0.0),
            reduction='none'
        )

    # Mask out invalid positions
    loss = loss * valid_mask.float()

    # Average over valid entries
    num_valid = valid_mask.float().sum()
    if num_valid == 0:
        return loss.sum() * 0.0  # Return 0 with gradient
    return loss.sum() / num_valid


def compute_class_weights(
    labels: np.ndarray,
    method: str = "balanced",
    min_weight: float = 0.1,
    max_weight: float = 10.0,
) -> np.ndarray:
    """Compute positive class weights for BCE loss from training labels.

    Args:
        labels: [N, 12] - binary labels (0 or 1), NaN for missing
        method: "balanced" (inverse frequency) or "effective_num"
        min_weight: minimum weight clamp
        max_weight: maximum weight clamp

    Returns:
        [12] - positive class weights per label

    IMPORTANT: Only use TRAINING DATA to compute weights.
    """
    if method == "balanced":
        # pos_weight = neg_count / pos_count
        pos_counts = np.nansum(labels, axis=0)
        total_counts = np.sum(~np.isnan(labels), axis=0)
        neg_counts = total_counts - pos_counts

        # Avoid division by zero
        weights = np.where(pos_counts > 0, neg_counts / np.maximum(pos_counts, 1), 1.0)

    elif method == "effective_num":
        # From "Class-Balanced Loss Based on Effective Number of Samples"
        # https://arxiv.org/abs/1901.05555
        beta = 0.9999
        pos_counts = np.nansum(labels, axis=0)
        effective_num = 1.0 - np.power(beta, pos_counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.mean() * labels.shape[1]  # Normalize

    else:
        raise ValueError(f"Unknown weight method: {method}")

    # Clamp weights
    weights = np.clip(weights, min_weight, max_weight)

    log.info(f"Computed class weights ({method}):")
    for i, (label, w) in enumerate(zip(LABELS, weights)):
        pos = np.nansum(labels[:, i])
        total = np.sum(~np.isnan(labels[:, i]))
        log.info(f"  {label}: pos={int(pos)}/{int(total)}, weight={w:.3f}")

    return weights


class MultiLabelBCELoss(nn.Module):
    """Multi-label BCE loss with configurable class weights."""

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return weighted_bce_loss(logits, targets, self.pos_weight)


def create_loss_function(
    train_labels: np.ndarray | None = None,
    weight_method: str = "balanced",
    device: torch.device = torch.device("cpu"),
) -> MultiLabelBCELoss:
    """Create loss function with class weights from training data.

    Args:
        train_labels: [N, 12] training labels for weight computation
        weight_method: method for computing weights
        device: device for weight tensor

    Returns:
        MultiLabelBCELoss instance
    """
    if train_labels is not None:
        weights = compute_class_weights(train_labels, method=weight_method)
        pos_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        pos_weight = None

    return MultiLabelBCELoss(pos_weight=pos_weight)