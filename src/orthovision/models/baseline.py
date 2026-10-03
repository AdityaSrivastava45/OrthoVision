"""OrthoVision Stage 3/4 models: DINOv2-S with mean pooling or label-specific attention MIL.

Stage 3 (mean pooling):
2.5D MRI [N, 3, H, W]
    ↓
DINOv2-S (shared across planes)
    ↓
per-sample embedding [N, D]
    ↓
mean pooling
    ↓
LayerNorm
    ↓
Linear classification head
    ↓
12 logits

Stage 4 (label-specific attention MIL):
2.5D MRI samples
       ↓
DINOv2-S
       ↓
sample embeddings H
       ↓
shared attention projection
       ↓
12 learned label/query vectors
       ↓
12 label-specific attention distributions
       ↓
12 label-specific study representations
       ↓
12 label-specific classifiers
       ↓
12 logits
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from orthovision.models.dinov2_encoder import DINOv2Config, DINOv2Encoder
from orthovision.models.pooling import PoolingModule, create_pooling_module
from orthovision.models.attention_mil import (
    LabelSpecificAttentionPooling,
    LabelSpecificClassifier,
    AttentionMILConfig,
)

log = logging.getLogger("orthovision.models.baseline")

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
class BaselineConfig:
    """Configuration for baseline model (supports both Stage 3 and Stage 4)."""
    # DINOv2 config
    dinov2: DINOv2Config = field(default_factory=DINOv2Config)

    # Pooling config
    pooling_type: str = "mean"  # "mean" | "attention" | "label_specific_attention"
    pooling_kwargs: dict = field(default_factory=dict)

    # Classifier config
    num_labels: int = 12
    classifier_dropout: float = 0.0

    # Attention MIL specific (only used when pooling_type == "label_specific_attention")
    attention_mil: Optional[AttentionMILConfig] = None

    def __post_init__(self):
        if self.pooling_kwargs is None:
            self.pooling_kwargs = {}
        if self.attention_mil is None and self.pooling_type == "label_specific_attention":
            self.attention_mil = AttentionMILConfig()


class DINOv2Baseline(nn.Module):
    """Baseline model for knee MRI multi-label classification.
    
    Supports both Stage 3 (mean pooling) and Stage 4 (label-specific attention MIL).
    """

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config
        self.labels = LABELS

        # DINOv2 encoder (shared across all planes)
        self.encoder = DINOv2Encoder(config.dinov2)
        embed_dim = config.dinov2.embed_dim

        # Pooling module
        if config.pooling_type == "label_specific_attention":
            # Pass attention MIL config via pooling_kwargs
            pooling_kwargs = dict(config.pooling_kwargs)
            if config.attention_mil is not None:
                pooling_kwargs["mil_config"] = config.attention_mil
        else:
            pooling_kwargs = config.pooling_kwargs

        self.pooling: PoolingModule = create_pooling_module(
            config.pooling_type,
            embed_dim,
            **pooling_kwargs,
        )

        # Classifier: different for mean pooling vs label-specific attention
        if config.pooling_type == "label_specific_attention":
            # Label-specific classifiers (one per label)
            self.classifier = LabelSpecificClassifier(
                embed_dim,
                config.num_labels,
                config.classifier_dropout,
            )
            self.norm = None  # LabelSpecificClassifier has its own LayerNorms
        else:
            # Shared LayerNorm + Linear classifier (Stage 3 baseline)
            self.norm = nn.LayerNorm(embed_dim)
            self.classifier = nn.Linear(embed_dim, config.num_labels)
            if config.classifier_dropout > 0:
                self.dropout = nn.Dropout(config.classifier_dropout)
            else:
                self.dropout = None

        log.info(f"Baseline model created:")
        log.info(f"  DINOv2 variant: {config.dinov2.variant}")
        log.info(f"  Embed dim: {embed_dim}")
        log.info(f"  Pooling: {config.pooling_type}")
        log.info(f"  Num labels: {config.num_labels}")
        log.info(f"  Frozen backbone: {config.dinov2.freeze_backbone}")
        if config.dinov2.freeze_backbone and config.dinov2.unfreeze_last_n_blocks > 0:
            log.info(f"  Unfrozen last {config.dinov2.unfreeze_last_n_blocks} blocks")

        self._log_trainable_params()

    def _log_trainable_params(self) -> None:
        """Log number of trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        log.info(f"  Total params: {total:,}")
        log.info(f"  Trainable params: {trainable:,}")
        log.info(f"  Frozen params: {frozen:,}")

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_metadata: list[list[dict]] | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            images: [B, N, 3, H, W] - batch of studies, each with N 2.5D samples
            mask: [B, N] - boolean mask for valid samples (True = valid)
            sample_metadata: List of per-study metadata for plane embeddings (required for attention MIL)
            return_attention: Whether to return attention weights (only for label_specific_attention)

        Returns:
            logits: [B, 12] - logits for each label
            attention_weights: [B, N, 12] - optional, only if return_attention=True and using attention MIL
        """
        B, N, C, H, W = images.shape

        # Flatten to [B*N, 3, H, W] for encoder
        images_flat = images.view(B * N, C, H, W)

        # Encode each 2.5D sample
        embeddings_flat = self.encoder(images_flat)  # [B*N, D]

        # Reshape back to [B, N, D]
        embeddings = embeddings_flat.view(B, N, -1)

        # Pool across samples to get study representation(s)
        if self.config.pooling_type == "label_specific_attention":
            # Label-specific attention returns [B, 12, D] study representations and [B, N, 12] attention
            study_representations, attention_weights = self.pooling(
                embeddings, mask, sample_metadata
            )
            # Classify each label's representation
            logits = self.classifier(study_representations)  # [B, 12]
        else:
            # Standard pooling (mean, attention) returns [B, D]
            study_embedding = self.pooling(embeddings, mask)  # [B, D]
            attention_weights = None
            # Classifier
            x = self.norm(study_embedding)
            if self.dropout is not None:
                x = self.dropout(x)
            logits = self.classifier(x)  # [B, 12]

        if return_attention and self.config.pooling_type == "label_specific_attention":
            return logits, attention_weights
        return logits

    def get_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Get per-sample embeddings without pooling/classification.

        Args:
            images: [B, N, 3, H, W]

        Returns:
            [B, N, D] - per-sample embeddings
        """
        B, N, C, H, W = images.shape
        images_flat = images.view(B * N, C, H, W)
        embeddings_flat = self.encoder(images_flat)
        return embeddings_flat.view(B, N, -1)

    def get_study_embedding(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_metadata: list[list[dict]] | None = None,
    ) -> torch.Tensor:
        """Get study-level embedding/representation(s) (after pooling, before classifier).

        Args:
            images: [B, N, 3, H, W]
            mask: [B, N] - boolean mask for valid samples
            sample_metadata: List of per-study metadata for plane embeddings

        Returns:
            If mean pooling: [B, D] - single study embedding
            If label_specific_attention: [B, 12, D] - one representation per label
        """
        B, N, C, H, W = images.shape
        images_flat = images.view(B * N, C, H, W)
        embeddings_flat = self.encoder(images_flat)
        embeddings = embeddings_flat.view(B, N, -1)

        if self.config.pooling_type == "label_specific_attention":
            study_representations, _ = self.pooling(embeddings, mask, sample_metadata)
            return study_representations
        else:
            return self.pooling(embeddings, mask)

    def get_attention_weights(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_metadata: list[list[dict]] | None = None,
    ) -> torch.Tensor | None:
        """Get attention weights for label-specific attention MIL.
        
        Args:
            images: [B, N, 3, H, W]
            mask: [B, N] - boolean mask for valid samples
            sample_metadata: List of per-study metadata for plane embeddings
            
        Returns:
            [B, N, 12] attention weights, or None if not using label_specific_attention
        """
        if self.config.pooling_type != "label_specific_attention":
            return None
        
        B, N, C, H, W = images.shape
        images_flat = images.view(B * N, C, H, W)
        embeddings_flat = self.encoder(images_flat)
        embeddings = embeddings_flat.view(B, N, -1)
        
        return self.pooling.get_attention_weights(embeddings, mask, sample_metadata)

    def inference(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_metadata: list[list[dict]] | None = None,
    ) -> dict:
        """Inference mode returning logits, attention weights, and metadata.
        
        Args:
            images: [B, N, 3, H, W] - batch of studies
            mask: [B, N] - boolean mask for valid samples
            sample_metadata: List of per-study metadata (preserved from dataset)
            
        Returns:
            Dict containing:
                - logits: [B, 12]
                - attention: [B, N, 12] or None
                - probs: [B, 12] - sigmoid probabilities
                - instance_metadata: List of per-instance metadata
        """
        self.eval()
        with torch.no_grad():
            if self.config.pooling_type == "label_specific_attention":
                logits, attention = self.forward(
                    images, mask, sample_metadata, return_attention=True
                )
            else:
                logits = self.forward(images, mask)
                attention = None
            
            probs = torch.sigmoid(logits)
        
        # Extract instance metadata if available
        instance_metadata = []
        if sample_metadata is not None:
            for study_meta in sample_metadata:
                # Keep all metadata fields for provenance
                instance_metadata.append(study_meta)
        
        return {
            "logits": logits.cpu(),
            "probs": probs.cpu(),
            "attention": attention.cpu() if attention is not None else None,
            "instance_metadata": instance_metadata,
        }


def create_baseline_model(config: BaselineConfig) -> DINOv2Baseline:
    """Factory function to create baseline model."""
    return DINOv2Baseline(config)