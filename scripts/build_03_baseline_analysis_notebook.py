"""Builds notebooks/03_baseline_analysis.ipynb from cell sources.

Rerun after editing any cell source:  python scripts/build_03_baseline_analysis_notebook.py
Keeps the notebook reproducible and diff-friendly in review.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

MD_INTRO = """\
# OrthoVision - Stage 3: DINOv2-S Baseline Analysis

**Objective.** Analyze the results of the DINOv2-S MRI-only baseline experiments.

**Scope.** This notebook visualizes training dynamics, per-label performance, and
embedding characteristics for the Stage 3 baseline model.

**Experiment Configuration:**
- Model: DINOv2-Small (ViT-S/14) frozen backbone
- Pooling: Mean pooling across 2.5D samples
- Classifier: LayerNorm + Linear (12 labels)
- Loss: BCEWithLogitsLoss with balanced class weights
- Input: 224×224 2.5D MRI triplets (3 channels)
- Training: 80/20 study-level split, patient-aware
"""

CODE_SETUP = """\
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

PROJECT = Path.cwd().resolve()
if PROJECT.name == "notebooks":
    PROJECT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from orthovision.data.tabular import LABELS

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")

print("Project:", PROJECT)
"""

CODE_LOAD_EXPERIMENT = """\
# Load experiment results
EXPERIMENT_DIR = PROJECT / "experiments" / "dinov2_baseline" / "dinov2_baseline_20260831_120000"  # Update with actual timestamp

config_path = EXPERIMENT_DIR / "config.json"
metrics_path = EXPERIMENT_DIR / "epoch_metrics.csv"
summary_path = EXPERIMENT_DIR / "summary.json"

with open(config_path) as f:
    config = json.load(f)

metrics_df = pd.read_csv(metrics_path)

with open(summary_path) as f:
    summary = json.load(f)

print(f"Experiment: {config['experiment_id']}")
print(f"Best macro AUC: {summary['best_macro_auc']:.4f} at epoch {summary['best_epoch']}")
print(f"Total time: {summary['total_time_seconds']:.1f}s")
print(f"Trainable params: {config['model']['trainable_params']:,}")
print(f"Total params: {config['model']['total_params']:,}")
"""

CODE_TRAINING_CURVES = """\
# Training/validation loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
axes[0].plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss', marker='o')
axes[0].plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Macro AUC curve
axes[1].plot(metrics_df['epoch'], metrics_df['val_macro_auc'], label='Val Macro AUC', marker='o', color='green')
axes[1].axhline(summary['best_macro_auc'], color='red', linestyle='--', label=f"Best: {summary['best_macro_auc']:.4f}")
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Macro ROC-AUC')
axes[1].set_title('Validation Macro AUC')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""

CODE_PER_LABEL_AUC = """\
# Per-label AUC bar chart
# Extract per-label AUCs from last epoch
last_epoch = metrics_df.iloc[-1]
per_label_auc = last_epoch.get('per_label_auc', {})

if isinstance(per_label_auc, str):
    per_label_auc = json.loads(per_label_auc)

if per_label_auc:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(per_label_auc.keys())
    aucs = list(per_label_auc.values())
    
    # Sort by AUC
    sorted_pairs = sorted(zip(labels, aucs), key=lambda x: x[1] if not np.isnan(x[1]) else -1)
    labels, aucs = zip(*sorted_pairs)
    
    colors = ['red' if np.isnan(a) else 'steelblue' for a in aucs]
    bars = ax.barh(labels, aucs, color=colors, edgecolor='black')
    
    ax.set_xlabel('ROC-AUC')
    ax.set_title('Per-Label ROC-AUC (Last Epoch)')
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        if not np.isnan(auc):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{auc:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()
else:
    print("Per-label AUC data not available in metrics")
"""

CODE_EMBEDDING_ANALYSIS = """\
# Embedding analysis
# Load predictions if available
preds_path = EXPERIMENT_DIR / "predictions" / "val_predictions.npz"

if preds_path.exists():
    data = np.load(preds_path, allow_pickle=False)
    study_uids = data['study_uids']
    logits = data['logits']
    probs = data['probs']
    labels = data['labels']
    
    print(f"Validation predictions shape: {probs.shape}")
    print(f"Number of studies: {len(study_uids)}")
    
    # Number of samples per study
    samples_per_study = []
    for uid in study_uids:
        # This would require access to the dataloader
        pass
    
    # Label prevalence
    print("\\nLabel prevalence in validation:")
    for i, label in enumerate(LABELS):
        valid = ~np.isnan(labels[:, i])
        if valid.sum() > 0:
            pos = labels[valid, i].sum()
            print(f"  {label}: {int(pos)}/{int(valid.sum())} = {pos/valid.sum():.2%}")
        else:
            print(f"  {label}: No valid labels")
else:
    print("Predictions file not found. Run evaluation first.")
"""

CODE_PCA_VISUALIZATION = """\
# PCA visualization of study embeddings (if available)
# This requires extracting embeddings from the model

try:
    from sklearn.decomposition import PCA
    
    # Check if embeddings are cached
    emb_path = EXPERIMENT_DIR / "embeddings_val.npz"
    
    if emb_path.exists():
        emb_data = np.load(emb_path)
        embeddings = emb_data['embeddings']  # [N_studies, embed_dim]
        study_labels = emb_data['study_uids']
        
        # PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot by each label
        for idx, label in enumerate(LABELS[:3]):  # First 3 labels
            ax = axes[idx]
            valid = ~np.isnan(labels[:, idx])
            if valid.sum() > 0:
                scatter = ax.scatter(embeddings_2d[valid, 0], embeddings_2d[valid, 1],
                                   c=labels[valid, idx], cmap='RdYlBu', alpha=0.7, s=30)
                ax.set_title(f'{label}')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                plt.colorbar(scatter, ax=ax)
        
        plt.suptitle(f'PCA of Study Embeddings (explained variance: {pca.explained_variance_ratio_.sum():.2%})')
        plt.tight_layout()
        plt.show()
    else:
        print("Embeddings not cached. Run with embedding extraction to enable PCA visualization.")
except ImportError:
    print("scikit-learn not available for PCA visualization")
"""

CODE_SUMMARY = """\
# Summary

This notebook provides visual analysis of the Stage 3 DINOv2-S baseline experiment.

**Key Metrics to Report:**
- Best validation macro AUC
- Per-label AUCs
- Training time
- Peak GPU memory
- Number of trainable parameters

**Next Steps for Stage 4:**
1. Implement attention-based pooling (MIL)
2. Add cross-view attention between planes
3. Experiment with partial fine-tuning (unfreeze last N blocks)
4. Add multimodal report encoder
5. Implement uncertainty estimation
"""

def build_notebook():
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": MD_INTRO.splitlines(keepends=True)
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CODE_SETUP.splitlines(keepends=True)
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CODE_LOAD_EXPERIMENT.splitlines(keepends=True)
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CODE_TRAINING_CURVES.splitlines(keepends=True)
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CODE_PER_LABEL_AUC.splitlines(keepends=True)
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CODE_EMBEDDING_ANALYSIS.splitlines(keepends=True)
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CODE_PCA_VISUALIZATION.splitlines(keepends=True)
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": CODE_SUMMARY.splitlines(keepends=True)
    })
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.6"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    out_path = PROJECT / "notebooks" / "03_baseline_analysis.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Written: {out_path}")

if __name__ == "__main__":
    build_notebook()