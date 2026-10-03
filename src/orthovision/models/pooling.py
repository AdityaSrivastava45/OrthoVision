"""Pooling modules for study-level aggregation.

The classification target belongs to the entire StudyInstanceUID.
Individual 2.5D samples are embedded by DINOv2, then aggregated
to produce a study-level representation.

For Stage 3 baseline, we use simple mean pooling only.
The module is designed to be replaceable for future experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from orthovision.models.dinov2_encoder import DINOv2Encoder


class PoolingModule(nn.Module):
    """Base class for pooling modules."""

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool embeddings to study representation.

        Args:
            embeddings: [B, N, D] - batch of study embeddings (N samples per study)
            mask: [B, N] - boolean mask for valid samples (True = valid)

        Returns:
            [B, D] - study-level representation
        """
        raise NotImplementedError


class MeanPooling(PoolingModule):
    """Simple mean pooling across samples.

    For variable N per study, uses mask to ignore padded positions.
    """

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean pool with optional mask.

        Args:
            embeddings: [B, N, D]
            mask: [B, N] - True for valid samples

        Returns:
            [B, D]
        """
        if mask is None:
            return embeddings.mean(dim=1)

        # mask: [B, N] -> [B, N, 1] for broadcasting
        mask = mask.unsqueeze(-1).float()  # [B, N, 1]
        masked_embeddings = embeddings * mask
        sums = masked_embeddings.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return sums / counts


class AttentionPooling(PoolingModule):
    """Attention-based pooling (for future use, not Stage 3 baseline).

    Uses a learnable query to attend over sample embeddings.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Attention pool with optional mask.

        Args:
            embeddings: [B, N, D]
            mask: [B, N] - True for valid samples (used as key_padding_mask)

        Returns:
            [B, D]
        """
        B = embeddings.shape[0]
        query = self.query.expand(B, -1, -1)  # [B, 1, D]

        # key_padding_mask: True = ignore (padding), False = attend
        if mask is not None:
            key_padding_mask = ~mask  # invert: True = valid -> False = attend
        else:
            key_padding_mask = None

        out, _ = self.attention(query, embeddings, embeddings,
                                key_padding_mask=key_padding_mask)
        return out.squeeze(1)  # [B, D]


def create_pooling_module(pooling_type: str, embed_dim: int, **kwargs) -> PoolingModule:
    """Factory function to create pooling module.

    Args:
        pooling_type: "mean" | "attention" | "label_specific_attention"
        embed_dim: embedding dimension
        **kwargs: additional arguments for specific pooling modules

    Returns:
        PoolingModule instance
    """
    if pooling_type == "mean":
        return MeanPooling()
    elif pooling_type == "attention":
        return AttentionPooling(embed_dim, **kwargs)
    elif pooling_type == "label_specific_attention":
        from orthovision.models.attention_mil import AttentionMILConfig, create_attention_mil_pooling
        mil_config = kwargs.get("mil_config") or AttentionMILConfig(**kwargs)
        return create_attention_mil_pooling(embed_dim, mil_config)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")