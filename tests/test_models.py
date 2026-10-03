"""Tests for Stage 3 model components."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from orthovision.models.baseline import DINOv2Baseline, BaselineConfig
from orthovision.models.dinov2_encoder import DINOv2Config, DINOv2Encoder
from orthovision.models.pooling import MeanPooling, AttentionPooling, create_pooling_module


def test_dinov2_encoder_creation():
    """Test DINOv2 encoder can be created."""
    config = DINOv2Config(
        variant="dinov2_vits14",
        pretrained_path=None,
        freeze_backbone=True,
        img_size=224,
    )
    encoder = DINOv2Encoder(config)
    
    assert encoder is not None
    assert encoder.embed_dim == 384
    assert encoder.config.freeze_backbone is True


def test_dinov2_encoder_forward():
    """Test DINOv2 encoder forward pass."""
    config = DINOv2Config(
        variant="dinov2_vits14",
        pretrained_path=None,
        freeze_backbone=True,
        img_size=224,
    )
    encoder = DINOv2Encoder(config)
    
    # Test with batch of 2.5D samples
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = encoder(x)
    
    assert out.shape == (4, 384)
    assert not torch.isnan(out).any()


def test_dinov2_encoder_frozen_params():
    """Test that frozen backbone parameters don't require grad."""
    config = DINOv2Config(
        variant="dinov2_vits14",
        pretrained_path=None,
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
    )
    encoder = DINOv2Encoder(config)
    
    # All params should be frozen
    for param in encoder.parameters():
        assert param.requires_grad is False


def test_dinov2_encoder_unfreeze_blocks():
    """Test unfreezing last N blocks."""
    config = DINOv2Config(
        variant="dinov2_vits14",
        pretrained_path=None,
        freeze_backbone=True,
        unfreeze_last_n_blocks=2,
    )
    encoder = DINOv2Encoder(config)
    
    # Some params should be trainable (the unfrozen blocks)
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    assert trainable_params > 0


def test_mean_pooling():
    """Test mean pooling module."""
    pooling = MeanPooling()
    
    # Test without mask
    embeddings = torch.randn(2, 5, 384)  # [B, N, D]
    out = pooling(embeddings)
    assert out.shape == (2, 384)
    assert torch.allclose(out[0], embeddings[0].mean(dim=0))
    
    # Test with mask
    mask = torch.tensor([[True, True, True, False, False],
                         [True, True, True, True, True]])
    out = pooling(embeddings, mask)
    assert out.shape == (2, 384)
    # First batch: mean of first 3
    expected = embeddings[0, :3].mean(dim=0)
    assert torch.allclose(out[0], expected, atol=1e-6)


def test_attention_pooling():
    """Test attention pooling module."""
    pooling = AttentionPooling(embed_dim=384, num_heads=4)
    
    embeddings = torch.randn(2, 5, 384)
    out = pooling(embeddings)
    assert out.shape == (2, 384)
    
    # Test with mask
    mask = torch.tensor([[True, True, True, False, False],
                         [True, True, True, True, True]])
    out = pooling(embeddings, mask)
    assert out.shape == (2, 384)


def test_create_pooling_module():
    """Test pooling module factory."""
    mean_pool = create_pooling_module("mean", 384)
    assert isinstance(mean_pool, MeanPooling)
    
    attn_pool = create_pooling_module("attention", 384, num_heads=4)
    assert isinstance(attn_pool, AttentionPooling)
    
    with pytest.raises(ValueError):
        create_pooling_module("unknown", 384)


def test_baseline_model_creation():
    """Test baseline model can be created."""
    config = BaselineConfig(
        dinov2=DINOv2Config(
            variant="dinov2_vits14",
            pretrained_path=None,
            freeze_backbone=True,
            img_size=224,
        ),
        pooling_type="mean",
        num_labels=12,
    )
    model = DINOv2Baseline(config)
    
    assert model is not None
    assert model.config.num_labels == 12


def test_baseline_model_forward():
    """Test baseline model forward pass."""
    config = BaselineConfig(
        dinov2=DINOv2Config(
            variant="dinov2_vits14",
            pretrained_path=None,
            freeze_backbone=True,
            img_size=224,
        ),
        pooling_type="mean",
        num_labels=12,
    )
    model = DINOv2Baseline(config)
    
    # Input: [B, N, 3, H, W]
    B, N = 2, 6
    images = torch.randn(B, N, 3, 224, 224)
    
    with torch.no_grad():
        logits = model(images)
    
    assert logits.shape == (B, 12)
    assert not torch.isnan(logits).any()


def test_baseline_model_with_mask():
    """Test baseline model with sample mask."""
    config = BaselineConfig(
        dinov2=DINOv2Config(
            variant="dinov2_vits14",
            pretrained_path=None,
            freeze_backbone=True,
            img_size=224,
        ),
        pooling_type="mean",
        num_labels=12,
    )
    model = DINOv2Baseline(config)
    
    B, N = 2, 6
    images = torch.randn(B, N, 3, 224, 224)
    mask = torch.tensor([
        [True, True, True, True, True, True],
        [True, True, True, False, False, False],
    ])
    
    with torch.no_grad():
        logits = model(images, mask)
    
    assert logits.shape == (B, 12)
    assert not torch.isnan(logits).any()


def test_baseline_model_get_embeddings():
    """Test getting per-sample embeddings."""
    config = BaselineConfig(
        dinov2=DINOv2Config(
            variant="dinov2_vits14",
            pretrained_path=None,
            freeze_backbone=True,
            img_size=224,
        ),
        pooling_type="mean",
        num_labels=12,
    )
    model = DINOv2Baseline(config)
    
    B, N = 2, 4
    images = torch.randn(B, N, 3, 224, 224)
    
    with torch.no_grad():
        embeddings = model.get_embeddings(images)
    
    assert embeddings.shape == (B, N, 384)


def test_baseline_model_get_study_embedding():
    """Test getting study-level embedding."""
    config = BaselineConfig(
        dinov2=DINOv2Config(
            variant="dinov2_vits14",
            pretrained_path=None,
            freeze_backbone=True,
            img_size=224,
        ),
        pooling_type="mean",
        num_labels=12,
    )
    model = DINOv2Baseline(config)
    
    B, N = 2, 4
    images = torch.randn(B, N, 3, 224, 224)
    
    with torch.no_grad():
        study_emb = model.get_study_embedding(images)
    
    assert study_emb.shape == (B, 384)


def test_baseline_parameter_count():
    """Test baseline model parameter counting."""
    config = BaselineConfig(
        dinov2=DINOv2Config(
            variant="dinov2_vits14",
            pretrained_path=None,
            freeze_backbone=True,
            img_size=224,
        ),
        pooling_type="mean",
        num_labels=12,
    )
    model = DINOv2Baseline(config)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    # DINOv2 ViT-Small has ~21M params, frozen
    # Classifier + pooling + norm ~ 4K params
    assert total > 20_000_000
    assert trainable < 10_000  # Only classifier head
    assert frozen > 20_000_000


def test_baseline_trainable_only_classifier():
    """Test that only classifier parameters are trainable in frozen mode."""
    config = BaselineConfig(
        dinov2=DINOv2Config(
            variant="dinov2_vits14",
            pretrained_path=None,
            freeze_backbone=True,
            img_size=224,
        ),
        pooling_type="mean",
        num_labels=12,
    )
    model = DINOv2Baseline(config)
    
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    
    # Should only have classifier and norm params trainable
    for name in trainable_names:
        assert "classifier" in name or "norm" in name or "pooling" in name


def test_baseline_different_pooling():
    """Test baseline with attention pooling."""
    config = BaselineConfig(
        dinov2=DINOv2Config(
            variant="dinov2_vits14",
            pretrained_path=None,
            freeze_backbone=True,
            img_size=224,
        ),
        pooling_type="attention",
        pooling_kwargs={"num_heads": 4},
        num_labels=12,
    )
    model = DINOv2Baseline(config)
    
    B, N = 2, 4
    images = torch.randn(B, N, 3, 224, 224)
    
    with torch.no_grad():
        logits = model(images)
    
    assert logits.shape == (B, 12)


def test_baseline_dropout():
    """Test baseline with classifier dropout."""
    config = BaselineConfig(
        dinov2=DINOv2Config(
            variant="dinov2_vits14",
            pretrained_path=None,
            freeze_backbone=True,
            img_size=224,
        ),
        pooling_type="mean",
        num_labels=12,
        classifier_dropout=0.1,
    )
    model = DINOv2Baseline(config)
    
    assert model.dropout is not None
    assert model.dropout.p == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])