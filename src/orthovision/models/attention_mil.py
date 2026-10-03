"""Label-Specific Attention Multiple Instance Learning for OrthoVision Stage 4.

Implements shared attention projection with 12 learned query vectors
for label-specific study representations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from orthovision.models.pooling import PoolingModule

log = logging.getLogger("orthovision.models.attention_mil")

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

PLANES = ["Sagittal", "Coronal", "Axial"]
PLANE_TO_IDX = {p: i for i, p in enumerate(PLANES)}


@dataclass
class AttentionMILConfig:
    """Configuration for Label-Specific Attention MIL module."""
    # Shared attention projection
    attention_hidden_dim: int = 128  # Dimension of shared attention features (z_i)
    
    # Query vectors
    num_labels: int = 12
    
    # Plane embeddings
    use_plane_embeddings: bool = True
    plane_embed_dim: int = 16  # Small relative to DINOv2 (384)
    num_planes: int = 3
    
    # Regularization (disabled by default, configurable for future)
    attention_temperature: float = 1.0
    dropout: float = 0.0
    entropy_regularization: float = 0.0
    
    # Output
    return_attention: bool = True  # Whether to return attention weights


class LabelSpecificAttentionPooling(PoolingModule):
    """Label-specific attention MIL pooling with shared projection and query vectors.
    
    Architecture:
    1. Shared attention projection: z_i = tanh(W * h_i') where h_i' = h_i + plane_emb(plane_i)
    2. 12 learned query vectors q_k (one per label)
    3. Label-specific attention: score_ik = q_k^T z_i
    4. Softmax across instances per label: attention_ik = softmax(score_ik)
    5. Label-specific study representation: v_k = sum_i attention_ik * h_i
    
    Returns:
    - study_representations: [B, 12, D] - one representation per label
    - attention_weights: [B, N, 12] - attention per instance per label (optional)
    """
    
    def __init__(
        self,
        embed_dim: int,
        config: Optional[AttentionMILConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.config = config or AttentionMILConfig()
        
        # Plane embeddings (small, shared)
        if self.config.use_plane_embeddings:
            self.plane_embedding = nn.Embedding(
                self.config.num_planes,
                self.config.plane_embed_dim,
            )
            # Project plane embedding to embed_dim for addition
            self.plane_proj = nn.Linear(self.config.plane_embed_dim, embed_dim)
        else:
            self.plane_embedding = None
            self.plane_proj = None
        
        # Shared attention projection: h_i' -> z_i
        self.attention_projection = nn.Sequential(
            nn.Linear(embed_dim, self.config.attention_hidden_dim),
            nn.Tanh(),
        )
        
        # 12 learned query vectors (one per label)
        self.query_vectors = nn.Parameter(
            torch.randn(self.config.num_labels, self.config.attention_hidden_dim) * 0.02
        )
        
        # Dropout for attention scores (optional regularization)
        if self.config.dropout > 0:
            self.attention_dropout = nn.Dropout(self.config.dropout)
        else:
            self.attention_dropout = None
        
        # Temperature scaling
        self.temperature = self.config.attention_temperature
        
        log.info(f"LabelSpecificAttentionPooling created:")
        log.info(f"  embed_dim: {embed_dim}")
        log.info(f"  attention_hidden_dim: {self.config.attention_hidden_dim}")
        log.info(f"  num_labels: {self.config.num_labels}")
        log.info(f"  use_plane_embeddings: {self.config.use_plane_embeddings}")
        log.info(f"  plane_embed_dim: {self.config.plane_embed_dim}")
        log.info(f"  attention_temperature: {self.temperature}")
        log.info(f"  dropout: {self.config.dropout}")

    def _get_plane_indices(self, plane_metadata: list[list[dict]]) -> torch.Tensor:
        """Extract plane indices from sample metadata.
        
        Args:
            plane_metadata: List of length B, each containing list of N sample metadata dicts
            
        Returns:
            [B, N] tensor of plane indices (0=Sagittal, 1=Coronal, 2=Axial)
        """
        B = len(plane_metadata)
        N = len(plane_metadata[0]) if B > 0 else 0
        plane_indices = torch.zeros(B, N, dtype=torch.long)
        
        for b in range(B):
            for n in range(N):
                meta = plane_metadata[b][n]
                # Try derived plane first, fallback to CSV plane
                plane = meta.get("plane_derived") or meta.get("plane_csv") or "Sagittal"
                plane_indices[b, n] = PLANE_TO_IDX.get(plane, 0)
        
        return plane_indices

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_metadata: list[list[dict]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for label-specific attention pooling.
        
        Args:
            embeddings: [B, N, D] - per-sample embeddings from DINOv2
            mask: [B, N] - boolean mask (True = valid sample)
            sample_metadata: List of per-study metadata for plane embeddings
            
        Returns:
            study_representations: [B, 12, D] - one study representation per label
            attention_weights: [B, N, 12] or None - attention per instance per label
        """
        B, N, D = embeddings.shape
        device = embeddings.device
        
        # Add plane embeddings if enabled
        if self.config.use_plane_embeddings and sample_metadata is not None:
            plane_indices = self._get_plane_indices(sample_metadata).to(device)  # [B, N]
            plane_emb = self.plane_embedding(plane_indices)  # [B, N, plane_embed_dim]
            plane_emb = self.plane_proj(plane_emb)  # [B, N, D]
            embeddings_with_plane = embeddings + plane_emb
        else:
            embeddings_with_plane = embeddings
        
        # Shared attention projection: z_i = tanh(W * h_i')
        # [B, N, D] -> [B, N, attention_hidden_dim]
        z = self.attention_projection(embeddings_with_plane)
        
        # Compute attention scores for each label
        # q_k: [12, attention_hidden_dim]
        # z: [B, N, attention_hidden_dim]
        # scores: [B, N, 12]
        scores = torch.einsum("bnd,ld->bnl", z, self.query_vectors)  # [B, N, 12]
        scores = scores / self.temperature
        
        # Apply mask (set invalid positions to -inf before softmax)
        if mask is not None:
            # mask: [B, N] -> [B, N, 1] for broadcasting
            mask_expanded = mask.unsqueeze(-1).float()  # [B, N, 1]
            scores = scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        # Softmax across instances (dim=1) for each label independently
        attention_weights = F.softmax(scores, dim=1)  # [B, N, 12]
        
        # Apply dropout to attention weights if enabled
        if self.attention_dropout is not None and self.training:
            attention_weights = self.attention_dropout(attention_weights)
            # Renormalize after dropout
            attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        # Compute label-specific study representations
        # v_k = sum_i attention_ik * h_i
        # attention_weights: [B, N, 12], embeddings: [B, N, D]
        # study_representations: [B, 12, D]
        study_representations = torch.einsum("bnl,bnd->bld", attention_weights, embeddings)
        
        # Verify attention normalization (debug)
        if self.training and torch.is_grad_enabled():
            # Check that attention sums to 1 per label (approximately)
            attn_sums = attention_weights.sum(dim=1)  # [B, 12]
            if not torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4):
                log.warning(f"Attention weights don't sum to 1: min={attn_sums.min():.4f}, max={attn_sums.max():.4f}")
        
        if self.config.return_attention:
            return study_representations, attention_weights
        else:
            return study_representations, None

    def get_attention_weights(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_metadata: list[list[dict]] | None = None,
    ) -> torch.Tensor:
        """Get attention weights without computing study representations.
        
        Useful for inference/analysis.
        
        Returns:
            [B, N, 12] attention weights
        """
        _, attention_weights = self.forward(embeddings, mask, sample_metadata)
        return attention_weights


class LabelSpecificClassifier(nn.Module):
    """Label-specific classification heads.
    
    Each label gets its own LayerNorm + Linear classifier.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_labels: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        
        # Per-label LayerNorm
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_labels)
        ])
        
        # Per-label linear classifiers
        self.classifiers = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_labels)
        ])
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, study_representations: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            study_representations: [B, 12, D] - one representation per label
            
        Returns:
            logits: [B, 12] - binary logits per label
        """
        B, K, D = study_representations.shape
        logits = []
        
        for k in range(K):
            x = study_representations[:, k, :]  # [B, D]
            x = self.layer_norms[k](x)
            if self.dropout is not None:
                x = self.dropout(x)
            logit_k = self.classifiers[k](x)  # [B, 1]
            logits.append(logit_k)
        
        return torch.cat(logits, dim=1)  # [B, 12]


def create_attention_mil_pooling(
    embed_dim: int,
    config: Optional[AttentionMILConfig] = None,
) -> LabelSpecificAttentionPooling:
    """Factory function to create attention MIL pooling module."""
    return LabelSpecificAttentionPooling(embed_dim, config)


def create_label_classifier(
    embed_dim: int,
    num_labels: int = 12,
    dropout: float = 0.0,
) -> LabelSpecificClassifier:
    """Factory function to create label-specific classifier."""
    return LabelSpecificClassifier(embed_dim, num_labels, dropout)


# ============================================================
# Attention Sanity Checks
# ============================================================

def check_attention_normalization(
    attention_weights: torch.Tensor,
    mask: torch.Tensor | None = None,
    tolerance: float = 1e-4,
) -> dict:
    """Check that attention weights sum to 1 per label per study.
    
    Args:
        attention_weights: [B, N, 12] attention weights
        mask: [B, N] boolean mask for valid samples
        tolerance: acceptable deviation from 1.0
        
    Returns:
        Dict with check results
    """
    B, N, K = attention_weights.shape
    device = attention_weights.device
    
    if mask is not None:
        # Only check valid positions
        mask_expanded = mask.unsqueeze(-1).float()  # [B, N, 1]
        masked_attention = attention_weights * mask_expanded
        sums = masked_attention.sum(dim=1)  # [B, 12]
    else:
        sums = attention_weights.sum(dim=1)  # [B, 12]
    
    # Check closeness to 1
    is_normalized = torch.allclose(sums, torch.ones_like(sums), atol=tolerance)
    max_deviation = (sums - 1.0).abs().max().item()
    min_sum = sums.min().item()
    max_sum = sums.max().item()
    
    return {
        "is_normalized": bool(is_normalized),
        "max_deviation": max_deviation,
        "min_sum": min_sum,
        "max_sum": max_sum,
        "per_study_per_label_sums": sums.cpu().numpy(),
    }


def check_attention_nan(attention_weights: torch.Tensor) -> dict:
    """Check for NaN or Inf values in attention weights.
    
    Returns:
        Dict with check results
    """
    has_nan = torch.isnan(attention_weights).any().item()
    has_inf = torch.isinf(attention_weights).any().item()
    nan_count = torch.isnan(attention_weights).sum().item()
    inf_count = torch.isinf(attention_weights).sum().item()
    
    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def check_variable_N_support(
    model: LabelSpecificAttentionPooling,
    embed_dim: int = 384,
    N_values: list[int] = [1, 2, 4, 6, 8, 12, 16],
) -> dict:
    """Test model with variable number of instances per study.
    
    Args:
        model: LabelSpecificAttentionPooling instance
        embed_dim: embedding dimension
        N_values: list of N (number of instances) to test
        
    Returns:
        Dict with test results
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        for N in N_values:
            try:
                embeddings = torch.randn(1, N, embed_dim)
                mask = torch.ones(1, N, dtype=torch.bool)
                study_repr, attention = model(embeddings, mask)
                
                # Check shapes
                assert study_repr.shape == (1, 12, embed_dim), f"Wrong study_repr shape: {study_repr.shape}"
                assert attention.shape == (1, N, 12), f"Wrong attention shape: {attention.shape}"
                
                # Check normalization
                norm_check = check_attention_normalization(attention, mask)
                nan_check = check_attention_nan(attention)
                
                results[N] = {
                    "success": True,
                    "study_repr_shape": tuple(study_repr.shape),
                    "attention_shape": tuple(attention.shape),
                    "normalization": norm_check,
                    "nan_check": nan_check,
                }
            except Exception as e:
                results[N] = {
                    "success": False,
                    "error": str(e),
                }
    
    return results


def check_masking_behavior(
    model: LabelSpecificAttentionPooling,
    embed_dim: int = 384,
    N: int = 8,
    num_valid: int = 4,
) -> dict:
    """Test that padded/masked instances receive zero attention.
    
    Args:
        model: LabelSpecificAttentionPooling instance
        embed_dim: embedding dimension
        N: total number of positions (including padding)
        num_valid: number of valid (non-padded) instances
        
    Returns:
        Dict with test results
    """
    model.eval()
    
    with torch.no_grad():
        embeddings = torch.randn(1, N, embed_dim)
        mask = torch.zeros(1, N, dtype=torch.bool)
        mask[0, :num_valid] = True
        
        study_repr, attention = model(embeddings, mask)
        
        # Check that padded positions have zero attention
        padded_attention = attention[0, num_valid:, :]  # [N - num_valid, 12]
        padded_sum = padded_attention.sum().item()
        max_padded_attention = padded_attention.max().item()
        
        # Valid positions should have non-zero attention
        valid_attention = attention[0, :num_valid, :]  # [num_valid, 12]
        valid_sum = valid_attention.sum().item()
        
        return {
            "padded_attention_sum": padded_sum,
            "max_padded_attention": max_padded_attention,
            "valid_attention_sum": valid_sum,
            "padded_effectively_zero": padded_sum < 1e-6,
            "attention_shape": tuple(attention.shape),
        }


def check_provenance_mapping(
    attention_weights: torch.Tensor,
    sample_metadata: list[list[dict]],
) -> dict:
    """Verify attention index maps to correct MRI instance metadata.
    
    Args:
        attention_weights: [B, N, 12] attention weights
        sample_metadata: List of per-study metadata
        
    Returns:
        Dict with verification results
    """
    B, N, K = attention_weights.shape
    issues = []
    
    for b in range(B):
        if len(sample_metadata[b]) != N:
            issues.append(f"Study {b}: metadata length {len(sample_metadata[b])} != N={N}")
            continue
        
        for n in range(N):
            meta = sample_metadata[b][n]
            # Check required fields exist
            required_fields = ["study_uid", "series_uid", "plane_derived", "center_slice_index"]
            for field in required_fields:
                if field not in meta:
                    issues.append(f"Study {b}, sample {n}: missing required field '{field}'")
    
    return {
        "provenance_intact": len(issues) == 0,
        "issues": issues,
        "num_studies_checked": B,
        "num_samples_per_study": N,
    }


def run_attention_sanity_checks(
    model: LabelSpecificAttentionPooling,
    embed_dim: int = 384,
    sample_metadata: list[list[dict]] | None = None,
) -> dict:
    """Run all attention sanity checks.
    
    Args:
        model: LabelSpecificAttentionPooling instance
        embed_dim: embedding dimension
        sample_metadata: Optional sample metadata for provenance check
        
    Returns:
        Dict with all check results
    """
    results = {}
    
    # 1. Variable N check
    results["variable_N"] = check_variable_N_support(model, embed_dim)
    
    # 2. Masking check
    results["masking"] = check_masking_behavior(model, embed_dim)
    
    # 3. Provenance check (if metadata provided)
    if sample_metadata is not None:
        with torch.no_grad():
            N = len(sample_metadata[0]) if sample_metadata else 6
            embeddings = torch.randn(1, N, embed_dim)
            mask = torch.ones(1, N, dtype=torch.bool)
            _, attention = model(embeddings, mask)
            results["provenance"] = check_provenance_mapping(attention, sample_metadata)
    
    # 4. Normalization and NaN check on a test forward pass
    with torch.no_grad():
        embeddings = torch.randn(2, 6, embed_dim)
        mask = torch.ones(2, 6, dtype=torch.bool)
        _, attention = model(embeddings, mask)
        results["normalization"] = check_attention_normalization(attention, mask)
        results["nan_check"] = check_attention_nan(attention)
    
    return results