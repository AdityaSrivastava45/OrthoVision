"""Tests for Stage 3 training components."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from orthovision.training.losses import weighted_bce_loss, compute_class_weights, create_loss_function
from orthovision.training.metrics import compute_auc_metrics, compute_macro_auc, predictions_to_numpy
from orthovision.training.split import create_study_split, verify_no_leakage, get_training_labels
from orthovision.training.trainer import collate_fn


def test_weighted_bce_loss():
    """Test weighted BCE loss computation."""
    logits = torch.randn(4, 12)
    targets = torch.randint(0, 2, (4, 12)).float()
    pos_weight = torch.ones(12)
    
    loss = weighted_bce_loss(logits, targets, pos_weight)
    
    assert loss.dim() == 0  # scalar
    assert loss >= 0
    assert not torch.isnan(loss)


def test_weighted_bce_loss_with_nan():
    """Test BCE loss handles NaN targets."""
    logits = torch.randn(4, 12)
    targets = torch.randint(0, 2, (4, 12)).float()
    targets[0, 0] = float('nan')
    targets[1, 5] = float('nan')
    pos_weight = torch.ones(12)
    
    loss = weighted_bce_loss(logits, targets, pos_weight)
    
    assert loss.dim() == 0
    assert loss >= 0
    assert not torch.isnan(loss)


def test_weighted_bce_loss_no_pos_weight():
    """Test BCE loss without pos_weight."""
    logits = torch.randn(4, 12)
    targets = torch.randint(0, 2, (4, 12)).float()
    
    loss = weighted_bce_loss(logits, targets, pos_weight=None)
    
    assert loss.dim() == 0
    assert loss >= 0


def test_compute_class_weights_balanced():
    """Test class weight computation with balanced method."""
    # Create synthetic labels with different prevalences
    labels = np.zeros((100, 12))
    labels[:50, 0] = 1   # 50% prevalence
    labels[:10, 1] = 1   # 10% prevalence
    labels[:5, 2] = 1    # 5% prevalence
    labels[:, 3] = np.nan  # All NaN
    
    weights = compute_class_weights(labels, method="balanced", max_weight=20.0)
    
    assert weights.shape == (12,)
    assert weights[0] == 1.0  # 50/50 = 1.0
    assert weights[1] == 9.0  # 90/10 = 9.0
    assert weights[2] == 19.0  # 95/5 = 19.0
    assert weights[3] == 1.0  # No valid labels -> default 1.0
    assert np.all(weights >= 0.1) and np.all(weights <= 20.0)


def test_compute_class_weights_effective_num():
    """Test class weight computation with effective_num method."""
    labels = np.zeros((100, 12))
    labels[:50, 0] = 1
    labels[:10, 1] = 1
    
    weights = compute_class_weights(labels, method="effective_num")
    
    assert weights.shape == (12,)
    assert np.all(weights >= 0.1) and np.all(weights <= 10.0)


def test_compute_class_weights_invalid_method():
    """Test class weight computation with invalid method."""
    labels = np.zeros((100, 12))
    
    with pytest.raises(ValueError):
        compute_class_weights(labels, method="invalid")


def test_create_loss_function():
    """Test loss function creation with class weights."""
    train_labels = np.zeros((100, 12))
    train_labels[:50, 0] = 1
    train_labels[:10, 1] = 1
    
    loss_fn = create_loss_function(train_labels, weight_method="balanced", device=torch.device("cpu"))
    
    assert loss_fn is not None
    assert loss_fn.pos_weight is not None
    assert loss_fn.pos_weight.shape == (12,)


def test_compute_auc_metrics():
    """Test AUC metrics computation."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, (100, 12)).astype(float)
    y_score = np.random.rand(100, 12)
    
    metrics = compute_auc_metrics(y_true, y_score)
    
    assert hasattr(metrics, 'per_label_auc')
    assert hasattr(metrics, 'macro_auc')
    assert hasattr(metrics, 'valid_labels')
    assert len(metrics.per_label_auc) == 12
    assert 0 <= metrics.macro_auc <= 1
    assert len(metrics.valid_labels) <= 12


def test_compute_auc_metrics_with_nan():
    """Test AUC metrics with NaN labels."""
    from orthovision.data.tabular import LABELS
    y_true = np.random.randint(0, 2, (100, 12)).astype(float)
    y_true[0, 0] = np.nan
    y_true[50, 5] = np.nan
    y_score = np.random.rand(100, 12)
    
    metrics = compute_auc_metrics(y_true, y_score)
    
    # First label should be in metrics (using actual label names)
    assert LABELS[0] in metrics.per_label_auc


def test_compute_auc_metrics_single_class():
    """Test AUC metrics with single-class labels."""
    from orthovision.data.tabular import LABELS
    y_true = np.zeros((100, 12))
    y_true[:, 0] = 1  # All positive for first label
    y_score = np.random.rand(100, 12)
    
    metrics = compute_auc_metrics(y_true, y_score)
    
    # First label should have NaN AUC (single class)
    assert np.isnan(metrics.per_label_auc[LABELS[0]])


def test_compute_macro_auc():
    """Test macro AUC computation."""
    per_label_auc = {
        'a': 0.8,
        'b': 0.9,
        'c': np.nan,
        'd': 0.7,
    }
    
    macro = compute_macro_auc(per_label_auc)
    
    # Mean of valid: (0.8 + 0.9 + 0.7) / 3 = 0.8
    assert abs(macro - 0.8) < 1e-6


def test_compute_macro_auc_all_nan():
    """Test macro AUC with all NaN."""
    per_label_auc = {'a': np.nan, 'b': np.nan}
    macro = compute_macro_auc(per_label_auc)
    assert macro == 0.0


def test_predictions_to_numpy():
    """Test logits to numpy conversion."""
    logits = torch.randn(4, 12)
    
    probs = predictions_to_numpy(logits, apply_sigmoid=True)
    
    assert probs.shape == (4, 12)
    assert np.all(probs >= 0) and np.all(probs <= 1)
    
    # Test without sigmoid
    raw = predictions_to_numpy(logits, apply_sigmoid=False)
    assert np.allclose(raw, logits.numpy())


def test_create_study_split():
    """Test study-level split creation."""
    study_ids = [f"study_{i}" for i in range(100)]
    config = type('Config', (), {
        'train_ratio': 0.8,
        'val_ratio': 0.2,
        'seed': 42,
        'patient_aware': False,
        'output_dir': 'splits'
    })()
    
    split = create_study_split(study_ids, config)
    
    assert len(split.train_study_ids) == 80
    assert len(split.val_study_ids) == 20
    assert set(split.train_study_ids).isdisjoint(set(split.val_study_ids))


def test_verify_no_leakage():
    """Test leakage verification."""
    train_ids = ["a", "b", "c"]
    val_ids = ["d", "e", "f"]
    
    verify_no_leakage(train_ids, val_ids)  # Should not raise
    
    # Test with leakage
    with pytest.raises(ValueError):
        verify_no_leakage(["a", "b", "c"], ["c", "d", "e"])


def test_get_training_labels():
    """Test getting training labels from split."""
    # This test needs a mock labels CSV
    import pandas as pd
    import tempfile
    
    LABELS = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
        "Fracture",
    ]
    
    # Create temp CSV with required columns
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame({
            "StudyInstanceUID": [f"study_{i}" for i in range(10)],
            "Report": ["report"] * 10,  # Required column
            **{label: np.random.randint(0, 2, 10) for label in LABELS}
        })
        df.to_csv(f.name, index=False)
        temp_path = f.name
    
    # Ensure file is fully written
    f.close()
    
    try:
        train_ids = [f"study_{i}" for i in range(5)]
        labels = get_training_labels(train_ids, temp_path)
        
        assert labels.shape == (5, 12)
        assert np.all((labels == 0) | (labels == 1))
    finally:
        os.unlink(temp_path)


def test_collate_fn():
    """Test collate function for variable-length samples."""
    batch = [
        {
            "images": torch.randn(4, 3, 224, 224),
            "sample_metadata": [],
            "labels": torch.randn(12),
            "study_uid": "study_1",
        },
        {
            "images": torch.randn(6, 3, 224, 224),
            "sample_metadata": [],
            "labels": torch.randn(12),
            "study_uid": "study_2",
        },
    ]
    
    collated = collate_fn(batch)
    
    assert collated["images"].shape == (2, 6, 3, 224, 224)
    assert collated["mask"].shape == (2, 6)
    assert collated["labels"].shape == (2, 12)
    assert len(collated["study_uids"]) == 2
    
    # Check mask
    assert collated["mask"][0].sum() == 4
    assert collated["mask"][1].sum() == 6


def test_collate_fn_single_item():
    """Test collate with single batch item."""
    batch = [
        {
            "images": torch.randn(3, 3, 224, 224),
            "sample_metadata": [],
            "labels": torch.randn(12),
            "study_uid": "study_1",
        },
    ]
    
    collated = collate_fn(batch)
    
    assert collated["images"].shape == (1, 3, 3, 224, 224)
    assert collated["mask"].shape == (1, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])