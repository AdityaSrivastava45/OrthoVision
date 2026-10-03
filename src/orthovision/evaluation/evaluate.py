"""Evaluation utilities for trained models."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from orthovision.models.baseline import DINOv2Baseline, BaselineConfig
from orthovision.models.dinov2_encoder import DINOv2Config
from orthovision.training.losses import create_loss_function
from orthovision.training.metrics import (
    AUCMetrics,
    compute_auc_metrics,
    log_metrics,
    predictions_to_numpy,
)
from orthovision.training.trainer import collate_fn

log = logging.getLogger("orthovision.evaluation.evaluate")


def _resolve_safe_path(base_dir: Path, relative_path: str) -> Path:
    """Resolve a path safely within base directory."""
    base_dir = base_dir.resolve()
    full_path = (base_dir / relative_path).resolve()
    if not str(full_path).startswith(str(base_dir)):
        raise ValueError(f"Path {full_path} escapes base directory {base_dir}")
    return full_path


def evaluate_model(
    model: DINOv2Baseline,
    dataloader: DataLoader,
    device: torch.device,
    criterion: Any = None,
    return_predictions: bool = False,
) -> tuple[float, AUCMetrics, dict[str, Any] | None]:
    """Evaluate model on a dataset."""
    model.eval()
    all_logits = []
    all_labels = []
    all_study_uids = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(images, mask)

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_study_uids.extend(batch["study_uids"])

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)

    y_score = predictions_to_numpy(logits_cat)
    y_true = labels_cat.numpy()
    metrics = compute_auc_metrics(y_true, y_score)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    predictions = None
    if return_predictions:
        predictions = {
            "study_uids": all_study_uids,
            "logits": logits_cat.numpy(),
            "probs": y_score,
            "labels": y_true,
        }

    return avg_loss, metrics, predictions


def evaluate_checkpoint(
    checkpoint_path: str,
    dataloader: DataLoader,
    device: torch.device,
    baseline_config: BaselineConfig | None = None,
    checkpoint_base_dir: str = "experiments/checkpoints",
) -> dict[str, Any]:
    """Evaluate a saved checkpoint."""
    base_dir = Path(checkpoint_base_dir).resolve()
    safe_path = _resolve_safe_path(base_dir, checkpoint_path)
    checkpoint = torch.load(safe_path, map_location=device, weights_only=True)

    if baseline_config is None:
        model_config = checkpoint.get("model_config", {})
        dinov2_config = DINOv2Config(**model_config.get("dinov2", {}))
        baseline_config = BaselineConfig(
            dinov2=dinov2_config,
            pooling_type=model_config.get("pooling_type", "mean"),
            num_labels=model_config.get("num_labels", 12),
        )

    model = DINOv2Baseline(baseline_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = create_loss_function(device=device)

    avg_loss, metrics, predictions = evaluate_model(
        model, dataloader, device, criterion, return_predictions=True
    )

    log_metrics(metrics, prefix="Checkpoint evaluation ")

    return {
        "loss": avg_loss,
        "metrics": metrics,
        "predictions": predictions,
        "checkpoint_info": {
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "best_macro_auc": checkpoint.get("best_macro_auc"),
        },
    }


def run_inference(
    model: DINOv2Baseline,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Run inference and return predictions with binary decisions."""
    model.eval()
    results = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            logits = model(images, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            for i, study_uid in enumerate(batch["study_uids"]):
                results.append({
                    "study_uid": study_uid,
                    "probabilities": dict(zip(model.labels, probs[i].tolist())),
                    "predictions": dict(zip(model.labels, preds[i].tolist())),
                })

    return results