"""Metrics for multi-label classification evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

log = logging.getLogger("orthovision.training.metrics")

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


@dataclass
class AUCMetrics:
    """Container for AUC metrics."""
    per_label_auc: dict[str, float]
    macro_auc: float
    valid_labels: list[str]
    per_label_pr_auc: dict[str, float] | None = None
    macro_pr_auc: float | None = None


def compute_per_label_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Compute ROC-AUC for each label.

    Args:
        y_true: [N, 12] - binary labels (0 or 1), NaN for missing
        y_score: [N, 12] - predicted probabilities or logits

    Returns:
        Dict mapping label name to AUC (or NaN if not computable)
    """
    aucs = {}
    for i, label in enumerate(LABELS):
        # Get valid (non-NaN) entries for this label
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() == 0:
            aucs[label] = np.nan
            continue

        y_true_label = y_true[mask, i]
        y_score_label = y_score[mask, i]

        # Check if both classes present
        unique_classes = np.unique(y_true_label)
        if len(unique_classes) < 2:
            aucs[label] = np.nan
            log.warning(f"Label '{label}' has only one class in validation: {unique_classes}")
            continue

        try:
            auc = roc_auc_score(y_true_label, y_score_label)
            aucs[label] = float(auc)
        except ValueError as e:
            log.warning(f"Could not compute AUC for {label}: {e}")
            aucs[label] = np.nan

    return aucs


def compute_per_label_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Compute PR-AUC for each label.

    Args:
        y_true: [N, 12] - binary labels (0 or 1), NaN for missing
        y_score: [N, 12] - predicted probabilities or logits

    Returns:
        Dict mapping label name to PR-AUC (or NaN if not computable)
    """
    pr_aucs = {}
    for i, label in enumerate(LABELS):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() == 0:
            pr_aucs[label] = np.nan
            continue

        y_true_label = y_true[mask, i]
        y_score_label = y_score[mask, i]

        unique_classes = np.unique(y_true_label)
        if len(unique_classes) < 2:
            pr_aucs[label] = np.nan
            continue

        try:
            pr_auc = average_precision_score(y_true_label, y_score_label)
            pr_aucs[label] = float(pr_auc)
        except ValueError as e:
            log.warning(f"Could not compute PR-AUC for {label}: {e}")
            pr_aucs[label] = np.nan

    return pr_aucs


def compute_macro_auc(per_label_auc: dict[str, float]) -> float:
    """Compute macro AUC (mean of valid per-label AUCs).

    Args:
        per_label_auc: dict from compute_per_label_auc

    Returns:
        Mean AUC across labels with valid AUC
    """
    valid_aucs = [v for v in per_label_auc.values() if not np.isnan(v)]
    if not valid_aucs:
        return 0.0
    return float(np.mean(valid_aucs))


def compute_auc_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    compute_pr_auc: bool = True,
) -> AUCMetrics:
    """Compute all AUC metrics.

    Args:
        y_true: [N, 12] - binary labels (0 or 1), NaN for missing
        y_score: [N, 12] - predicted probabilities or logits
        compute_pr_auc: whether to compute PR-AUC

    Returns:
        AUCMetrics object
    """
    per_label_auc = compute_per_label_auc(y_true, y_score)
    macro_auc = compute_macro_auc(per_label_auc)
    valid_labels = [k for k, v in per_label_auc.items() if not np.isnan(v)]

    per_label_pr_auc = None
    macro_pr_auc = None
    if compute_pr_auc:
        per_label_pr_auc = compute_per_label_pr_auc(y_true, y_score)
        valid_pr = [v for v in per_label_pr_auc.values() if not np.isnan(v)]
        macro_pr_auc = float(np.mean(valid_pr)) if valid_pr else 0.0

    return AUCMetrics(
        per_label_auc=per_label_auc,
        macro_auc=macro_auc,
        valid_labels=valid_labels,
        per_label_pr_auc=per_label_pr_auc,
        macro_pr_auc=macro_pr_auc,
    )


def log_metrics(metrics: AUCMetrics, prefix: str = "") -> None:
    """Log metrics in a readable format."""
    log.info(f"{prefix}Per-label ROC-AUC:")
    for label in LABELS:
        auc = metrics.per_label_auc.get(label, np.nan)
        if np.isnan(auc):
            log.info(f"  {label}: N/A (insufficient data)")
        else:
            log.info(f"  {label}: {auc:.4f}")

    log.info(f"{prefix}Macro ROC-AUC: {metrics.macro_auc:.4f} "
             f"({len(metrics.valid_labels)}/12 labels valid)")

    if metrics.per_label_pr_auc is not None:
        log.info(f"{prefix}Per-label PR-AUC:")
        for label in LABELS:
            pr_auc = metrics.per_label_pr_auc.get(label, np.nan)
            if np.isnan(pr_auc):
                log.info(f"  {label}: N/A")
            else:
                log.info(f"  {label}: {pr_auc:.4f}")
        log.info(f"{prefix}Macro PR-AUC: {metrics.macro_pr_auc:.4f}")


def predictions_to_numpy(
    logits: torch.Tensor,
    apply_sigmoid: bool = True,
) -> np.ndarray:
    """Convert model logits to numpy probabilities.

    Args:
        logits: [B, 12] tensor
        apply_sigmoid: whether to apply sigmoid

    Returns:
        [B, 12] numpy array
    """
    with torch.no_grad():
        if apply_sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = logits
        return probs.cpu().numpy()