"""Builds notebooks/02_2p5d_visualization.ipynb from cell sources.

Rerun after editing any cell source:  python scripts/build_2p5d_visualization_notebook.py
Keeps the notebook reproducible and diff-friendly in review.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

MD_INTRO = """\
# OrthoVision - Stage 2: 2.5D Preprocessing Visual Validation

**Objective.** Visually verify that the anatomically correct MRI → 2.5D pipeline produces
correctly ordered, normalized, and sampled image triplets suitable for DINoV2-style backbones.

**Scope.** This notebook processes a small number of studies from the on-disk subset
(`data/raw/train_series/`) and visualizes every stage of the pipeline:

1. Raw DICOM slice (with orientation/LPS metadata)
2. Ordered series montage (all slices in anatomical order)
3. Selected anatomical sample locations (6 per plane by default)
4. Individual 2.5D triplets (previous / center / next channels)
5. Final resized model input (224×224 letterbox)

**Provenance & rules.**
- Raw data under `data/raw/` is **read-only**; this notebook writes nothing into it.
- Figures are deterministic: studies/series are picked by sorted UID order.
- Every visualization shows the complete provenance metadata.
- **No neural network is implemented or trained here** — this is pure preprocessing validation.
"""

CODE_SETUP = """\
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

PROJECT = Path.cwd().resolve()
if PROJECT.name == "notebooks":
    PROJECT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from orthovision.data.tabular import load_series
from orthovision.data.pixels import decode_path, display_preview
from orthovision.preprocessing.config import load_preprocess_config
from orthovision.preprocessing.discovery import SeriesInfo, discover_studies
from orthovision.preprocessing.pipeline import StudyPreprocessor
from orthovision.preprocessing.selection import select_study_series

RAW = PROJECT / "data" / "raw"
REPORTS = PROJECT / "outputs" / "reports"
HEADERS_CSV = REPORTS / "dicom_headers.csv"
SERIES_CSV = RAW / "train_series.csv"

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 9

print("project:", PROJECT)
print("headers cache exists:", HEADERS_CSV.exists())
"""

CODE_LOAD = """\
if not HEADERS_CSV.exists():
    raise RuntimeError(
        "Header cache missing. Run first:  python scripts/scan_dicom_headers.py"
    )

cfg = load_preprocess_config()
print("Config fingerprint:", cfg.fingerprint())
print("Image size:", cfg.image_size)
print("Planes:", cfg.planes)
print("Samples per plane:", cfg.num_samples_per_plane)
print("Normalization:", cfg.normalization_method, cfg.normalization_scope)
print("Neighbor strategy:", cfg.neighbor_strategy, cfg.neighbor_offsets)
print("Mode:", cfg.mode)

series_csv = load_series(SERIES_CSV)
studies = discover_studies(HEADERS_CSV, SERIES_CSV, cfg)
study_ids = sorted(studies.keys())
print(f"Discovered {len(study_ids)} studies on disk")

# Pick first few studies for visualization
vis_studies = study_ids[:3]  # first 3 studies
print(f"Visualizing studies: {vis_studies}")
"""

CODE_VISUALIZE_STUDY = """\
def visualize_study(study_uid: str, studies: dict, series_csv: pd.DataFrame, cfg):
    \"\"\"Process one study and visualize all pipeline stages.\"\"\"
    series_infos = studies[study_uid]
    
    # 1. Print study overview
    print(f"\\n{'='*60}")
    print(f"STUDY: {study_uid}")
    print(f"{'='*60}")
    for si in series_infos:
        print(f"  Series {si.series_uid[-12:]:>12} | Plane CSV={si.plane_csv} Derived={si.plane_derived} "
              f"FS={si.fluid_sensitive} Slices={si.n_slices} {si.rows}x{si.columns} "
              f"spacing={si.pixel_spacing_mm} ordering={si.ordering_method}")
    
    # 2. Series selection
    selections = select_study_series(series_infos, cfg)
    print("\\n  Selection report:")
    for plane in cfg.planes:
        report = selections.get(plane)
        if report and report.selected_series_uid:
            sel = report.selected
            print(f"    {plane}: SELECTED {report.selected_series_uid[-12:]} "
                  f"(score={sel.score:.3f})")
            for c in report.candidates:
                print(f"      Candidate {c.series_uid[-12:]} score={c.score:.3f} "
                      f"[{', '.join(c.reasons[:3])}]")
        else:
            print(f"    {plane}: NO CANDIDATE")
    
    # 3. Process through pipeline
    prep = StudyPreprocessor(cfg, data_root=RAW)
    result = prep.process_study(study_uid, series_infos)
    
    print(f"\\n  Generated {len(result.samples)} 2.5D samples")
    if result.skipped_series:
        for skip in result.skipped_series:
            print(f"  SKIPPED: {skip}")
    
    # 4. For each selected series, visualize
    for plane in cfg.planes:
        report = selections.get(plane)
        if not report or not report.selected_series_uid:
            continue
        info = next(s for s in series_infos if s.series_uid == report.selected_series_uid)
        visualize_series(info, result, plane, study_uid)
    
    return result

def visualize_series(info: SeriesInfo, result: 'StudyResult', plane: str, study_uid: str):
    \"\"\"Visualize one series through the pipeline stages.\"\"\"
    samples = [s for s in result.samples if s.metadata.get('plane_csv') == plane]
    if not samples:
        print(f"  No samples for {plane}")
        return
    
    print(f"\\n  --- {plane} Series {info.series_uid[-12:]} ---")
    print(f"  Slices: {info.n_slices}, Ordering: {info.ordering_method}")
    print(f"  Laterality: {info.laterality}, PatientPosition: {info.patient_position}")
    print(f"  Normal (LPS): {info.orientation_normal}")
    
    # --- VISUALIZATION 1: Raw DICOM slice ---
    visualize_raw_slice(info, plane, study_uid)
    
    # --- VISUALIZATION 2: Ordered series montage ---
    visualize_ordered_montage(info, plane, study_uid)
    
    # --- VISUALIZATION 3: Selected sample locations ---
    visualize_sample_locations(info, samples, plane, study_uid)
    
    # --- VISUALIZATION 4: Individual 2.5D triplets ---
    visualize_triplets(samples, plane, study_uid)
    
    # --- VISUALIZATION 5: Final resized model input ---
    visualize_final_input(samples, plane, study_uid)

def visualize_raw_slice(info: SeriesInfo, plane: str, study_uid: str):
    \"\"\"Show a single raw DICOM slice with metadata.\"\"\"    
    mid_idx = info.n_slices // 2
    row = info.slices_ordered[mid_idx]
    path = RAW / row.path
    payload = decode_path(path)
    preview = display_preview(payload, apply_window=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(preview, cmap='gray')
    ax.set_title(
        f"Raw DICOM Slice (Instance #{row.instance_number})\n"
        f"Study: {study_uid[-12:]} | Series: {info.series_uid[-12:]} | Plane: {plane}\n"
        f"SOP: {row.sop_uid[-12:]} | Size: {row.rows}x{row.columns} | "
        f"Spacing: {row.pixel_spacing} | SlicePos: {row.extra.get('slice_thickness')}\n"
        f"RescaleSlope: {row.rescale_slope} | Photometric: {row.photometric}",
        fontsize=8
    )
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_ordered_montage(info: SeriesInfo, plane: str, study_uid: str):
    \"\"\"Show all slices in anatomical order as a montage.\"\"\"
    n_show = min(info.n_slices, 24)
    indices = np.linspace(0, info.n_slices - 1, n_show, dtype=int)
    
    n_cols = 6
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_show > 1 else [axes]
    
    for i, idx in enumerate(indices):
        row = info.slices_ordered[idx]
        path = RAW / row.path
        payload = decode_path(path)
        preview = display_preview(payload, apply_window=True)
        axes[i].imshow(preview, cmap='gray')
        pos = info.geometry.positions_mm[idx] if info.geometry.positions_mm[idx] is not None else '?'
        axes[i].set_title(f"Idx {idx} (Inst#{row.instance_number})\nPos: {pos:.2f}mm", fontsize=7)
        axes[i].axis('off')
    
    for i in range(n_show, len(axes)):
        axes[i].axis('off')
    
    fig.suptitle(
        f"Ordered Series Montage — {plane} | Study: {study_uid[-12:]} | "
        f"Series: {info.series_uid[-12:]} | {info.n_slices} slices | "
        f"Ordering: {info.ordering_method}",
        fontsize=10
    )
    plt.tight_layout()
    plt.show()

def visualize_sample_locations(info: SeriesInfo, samples: list, plane: str, study_uid: str):
    \"\"\"Show where the 6 sample locations fall in the ordered series.\"\"\"
    positions = info.geometry.positions_mm
    valid_positions = [p for p in positions if p is not None]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    
    # Plot all slice positions
    if valid_positions:
        ax.scatter(range(len(valid_positions)), valid_positions, s=30, alpha=0.5, label='All slices')
    
    # Highlight sample centers
    center_indices = [s.metadata['center_slice_index'] for s in samples]
    center_positions = [positions[i] if positions[i] is not None else 0 for i in center_indices]
    ax.scatter(center_indices, center_positions, s=100, c='red', marker='*', zorder=5,
               label=f'Sample centers (n={len(samples)})')
    
    ax.set_xlabel('Slice index (anatomically ordered)')
    ax.set_ylabel('Physical position (mm)')
    ax.set_title(f'Sample Locations — {plane} | Study: {study_uid[-12:]} | Series: {info.series_uid[-12:]}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print sample details
    print(f"  Sample locations for {plane}:")
    for s in samples:
        meta = s.metadata
        print(f"    Center idx={meta['center_slice_index']} | "
              f"Slice indices={meta['slice_indices']} | "
              f"Positions={['{:.2f}'.format(p) if p is not None else '?' for p in meta['physical_positions_mm']]}")

def visualize_triplets(samples: list, plane: str, study_uid: str):
    \"\"\"Show individual 2.5D triplets as 3-channel visualization.\"\"\"
    n_show = min(len(samples), 6)
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, 3)
    
    for i, sample in enumerate(samples[:n_show]):
        meta = sample.metadata
        img = sample.image  # [3, H, W]
        
        for ch in range(3):
            ax = axes[i, ch]
            ax.imshow(img[ch], cmap='gray', vmin=0, vmax=1)
            role = meta['channel_roles'][ch]
            sop = meta['sop_instance_uids'][ch][-12:]
            pos = meta['physical_positions_mm'][ch]
            ax.set_title(f"{role.capitalize()}\nSOP:{sop} Pos:{pos:.2f}mm" if pos else f"{role.capitalize()}\nSOP:{sop}",
                         fontsize=8)
            ax.axis('off')
    
    fig.suptitle(f'2.5D Triplets — {plane} | Study: {study_uid[-12:]}', fontsize=10)
    plt.tight_layout()
    plt.show()

def visualize_final_input(samples: list, plane: str, study_uid: str):
    \"\"\"Show final model-ready input tensor.\"\"\"
    n_show = min(len(samples), 6)
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    if n_show == 1:
        axes = [axes]
    
    for i, sample in enumerate(samples[:n_show]):
        img = sample.image  # [3, H, W]
        # Convert to RGB-like for display
        rgb = np.transpose(img, (1, 2, 0))  # [H, W, 3]
        axes[i].imshow(rgb, vmin=0, vmax=1)
        meta = sample.metadata
        axes[i].set_title(
            f"Sample {i+1}\nCenter idx={meta['center_slice_index']}\n"
            f"Indices={meta['slice_indices']}\n"
            f"Size={img.shape[1:]}",
            fontsize=8
        )
        axes[i].axis('off')
    
    fig.suptitle(f'Final Model Input (3ch, {samples[0].image.shape[1]}x{samples[0].image.shape[2]}) — {plane} | Study: {study_uid[-12:]}', fontsize=10)
    plt.tight_layout()
    plt.show()

# Process and visualize each study
for study_uid in vis_studies:
    visualize_study(study_uid, studies, series_csv, cfg)
"""

CODE_SUMMARY = """\
# Summary

This notebook has demonstrated the complete Stage-2 preprocessing pipeline:

1. **Series Discovery** — Robust grouping by StudyInstanceUID → SeriesInstanceUID → slices
2. **Physical Slice Ordering** — Using ImageOrientationPatient × ImagePositionPatient (LPS coordinates)
3. **Series Selection** — Configurable ranking by fluid-sensitivity, slice count, resolution, metadata completeness, geometric consistency
4. **Intensity Normalization** — Percentile (0.5/99.5) clipping → [0,1] scaling; handles MONOCHROME1 inversion
5. **Physical-Space Sampling** — 6 evenly-spaced anatomical locations per plane (avoiding boundaries)
6. **2.5D Triplet Construction** — Neighboring slices (index offsets -1, 0, +1) as 3 channels
7. **Letterbox Resize** — Aspect-preserving resize to 224×224 with centered padding

**All metadata is preserved** for every generated 2.5D sample, enabling full provenance
back to the original DICOM slices.

**Next Steps for Stage 3:**
- Experiment with neighbor strategy: `physical_distance` instead of `index` offsets
- Try `normalization_scope=triplet` to preserve relative intensity across channels
- Adjust `boundary_margin`, `num_samples_per_plane`, `sample_positions` for different anatomies
- Tune `ranking_weights` for series selection based on downstream model performance
- Evaluate `zscore` normalization vs `percentile` for DINOv2 backbones
"""

def build_notebook():
    cells = []
    
    # Markdown intro
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": MD_INTRO.splitlines(keepends=True)
    })
    
    # Setup code
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CODE_SETUP.splitlines(keepends=True)
    })
    
    # Load data
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CODE_LOAD.splitlines(keepends=True)
    })
    
    # Visualization functions
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CODE_VISUALIZE_STUDY.splitlines(keepends=True)
    })
    
    # Summary
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
    
    out_path = PROJECT / "notebooks" / "02_2p5d_visualization.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Written: {out_path}")

if __name__ == "__main__":
    build_notebook()