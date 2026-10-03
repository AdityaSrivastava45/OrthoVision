"""Training loop for OrthoVision baseline model."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from orthovision.models.baseline import DINOv2Baseline
from orthovision.training.losses import create_loss_function, MultiLabelBCELoss
from orthovision.training.metrics import AUCMetrics, compute_auc_metrics, log_metrics, predictions_to_numpy

log = logging.getLogger("orthovision.training.trainer")


@dataclass
class TrainerConfig:
    """Configuration for training."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 4  # Per GPU, effective = batch_size * grad_accum
    grad_accum_steps: int = 1

    # Mixed precision
    use_amp: bool = True  # Automatic mixed precision

    # Scheduler
    scheduler: str = "cosine"  # "cosine" | "constant" | "reduce_on_plateau"
    warmup_epochs: int = 2

    # Checkpointing
    checkpoint_dir: str = "experiments/checkpoints"
    save_best: bool = True
    save_last: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 1  # Evaluate every N epochs

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Reproducibility
    deterministic: bool = False
    seed: int = 42


def collate_fn(batch: list[dict]) -> dict:
    """Collate function for variable-length study samples.

    Each item in batch:
        {
            "images": [N_i, 3, H, W],
            "sample_metadata": [...],
            "labels": [12],
            "study_uid": str
        }

    Returns padded batch:
        {
            "images": [B, max_N, 3, H, W],
            "mask": [B, max_N] - True for valid samples,
            "labels": [B, 12],
            "study_uids": [B],
            "sample_metadata": [B][max_N] - list of per-instance metadata
        }
    """
    # Find max number of samples in this batch
    max_n = max(item["images"].shape[0] for item in batch)
    B = len(batch)
    C, H, W = batch[0]["images"].shape[1:]

    # Pad images
    images_padded = torch.zeros(B, max_n, C, H, W, dtype=batch[0]["images"].dtype)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    labels = torch.zeros(B, 12, dtype=torch.float32)
    study_uids = []
    sample_metadata = []

    for i, item in enumerate(batch):
        n = item["images"].shape[0]
        images_padded[i, :n] = item["images"]
        mask[i, :n] = True
        labels[i] = item["labels"]
        study_uids.append(item["study_uid"])
        # Pad metadata with empty dicts for missing samples
        meta = item["sample_metadata"]
        padded_meta = meta + [{}] * (max_n - n)
        sample_metadata.append(padded_meta)

    return {
        "images": images_padded,
        "mask": mask,
        "labels": labels,
        "study_uids": study_uids,
        "sample_metadata": sample_metadata,
    }


class Trainer:
    """Trainer for DINOv2 baseline model."""

    def __init__(
        self,
        model: DINOv2Baseline,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        experiment_logger: Any = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.experiment_logger = experiment_logger

        # Loss function (will be set with class weights)
        self.criterion: MultiLabelBCELoss | None = None

        # Optimizer (only trainable params)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp) if device.type == "cuda" else None

        # State
        self.epoch = 0
        self.global_step = 0
        self.best_macro_auc = 0.0
        self.best_epoch = 0
        self.early_stop_counter = 0

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_auc": [],
            "val_per_label_auc": [],
            "learning_rate": [],
        }

        log.info(f"Trainer initialized on {device}")
        log.info(f"  Batch size: {config.batch_size}, Grad accum: {config.grad_accum_steps}")
        log.info(f"  Effective batch size: {config.batch_size * config.grad_accum_steps}")
        log.info(f"  AMP: {config.use_amp}")
        log.info(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=self.config.learning_rate * 0.001,
            )
        elif self.config.scheduler == "constant":
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def set_criterion(self, train_labels: np.ndarray) -> None:
        """Set loss function with class weights computed from training data."""
        self.criterion = create_loss_function(
            train_labels=train_labels,
            weight_method="balanced",
            device=self.device,
        )
        log.info("Loss function configured with training class weights")

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            sample_metadata = batch.get("sample_metadata")

            # Forward pass with AMP
            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images, mask, sample_metadata)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.config.grad_accum_steps

                self.scaler.scale(loss).backward()
            else:
                logits = self.model(images, mask, sample_metadata)
                loss = self.criterion(logits, labels)
                loss = loss / self.config.grad_accum_steps
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.config.grad_accum_steps
            num_batches += 1

            # Logging
            if batch_idx % self.config.log_interval == 0:
                log.info(f"Epoch {self.epoch} [{batch_idx}/{len(self.train_loader)}] "
                         f"Loss: {loss.item() * self.config.grad_accum_steps:.4f} "
                         f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def evaluate(self) -> tuple[float, AUCMetrics]:
        """Evaluate on validation set."""
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            images = batch["images"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            sample_metadata = batch.get("sample_metadata")

            if self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images, mask, sample_metadata)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images, mask, sample_metadata)
                loss = self.criterion(logits, labels)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            num_batches += 1

        # Concatenate all predictions
        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)

        # Compute metrics
        y_score = predictions_to_numpy(logits_cat)
        y_true = labels_cat.numpy()
        metrics = compute_auc_metrics(y_true, y_score)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, metrics

    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_macro_auc": self.best_macro_auc,
            "best_epoch": self.best_epoch,
            "config": asdict(self.config),
            "model_config": {
                "dinov2": self.model.config.dinov2.__dict__,
                "pooling_type": self.model.config.pooling_type,
                "num_labels": self.model.config.num_labels,
            },
        }

        torch.save(checkpoint, path)
        log.info(f"Checkpoint saved: {path} (best={is_best})")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_macro_auc = checkpoint["best_macro_auc"]
        self.best_epoch = checkpoint["best_epoch"]
        log.info(f"Checkpoint loaded: {path} (epoch {self.epoch}, best AUC {self.best_macro_auc:.4f})")

    def train(self) -> dict[str, Any]:
        """Run full training loop."""
        log.info("Starting training...")
        start_time = time.time()

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            # Validate
            if (epoch + 1) % self.config.eval_interval == 0:
                val_loss, metrics = self.evaluate()
                self.history["val_loss"].append(val_loss)
                self.history["val_macro_auc"].append(metrics.macro_auc)
                self.history["val_per_label_auc"].append(metrics.per_label_auc)

                log_metrics(metrics, prefix=f"Epoch {epoch} ")

                # Checkpointing
                is_best = metrics.macro_auc > self.best_macro_auc + self.config.early_stopping_min_delta
                if is_best:
                    self.best_macro_auc = metrics.macro_auc
                    self.best_epoch = epoch
                    self.early_stop_counter = 0
                    if self.config.save_best:
                        self.save_checkpoint(
                            Path(self.config.checkpoint_dir) / "best_model.pt",
                            is_best=True,
                        )
                else:
                    self.early_stop_counter += 1

                # Scheduler step
                if self.config.scheduler == "reduce_on_plateau":
                    self.scheduler.step(metrics.macro_auc)
                elif self.config.scheduler == "cosine" and epoch >= self.config.warmup_epochs:
                    self.scheduler.step()
                elif self.config.scheduler == "constant":
                    self.scheduler.step()

                # Log to experiment tracker
                if self.experiment_logger:
                    self.experiment_logger.log_metrics({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_macro_auc": metrics.macro_auc,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    })

            # Save last checkpoint
            if self.config.save_last and (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt",
                )

            # Early stopping
            if self.early_stop_counter >= self.config.early_stopping_patience:
                log.info(f"Early stopping triggered after {epoch} epochs")
                break

            epoch_time = time.time() - epoch_start
            log.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")

        total_time = time.time() - start_time
        log.info(f"Training completed in {total_time:.1f}s")
        log.info(f"Best macro AUC: {self.best_macro_auc:.4f} at epoch {self.best_epoch}")

        return {
            "best_macro_auc": self.best_macro_auc,
            "best_epoch": self.best_epoch,
            "total_time": total_time,
            "history": self.history,
        }