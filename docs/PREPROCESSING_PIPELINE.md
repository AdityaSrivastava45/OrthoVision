# OrthoVision Stage 2 — Preprocessing Pipeline Documentation

**Project:** OrthoVision: Explainable Multimodal AI for Knee MRI Analysis  
**Stage:** 2 — Anatomically Correct MRI → 2.5D Pipeline  
**Date:** 2026-08-31  
**Version:** 1.0

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [DICOM Loading](#2-dicom-loading)
3. [Geometry Reconstruction](#3-geometry-reconstruction)
4. [Slice Ordering](#4-slice-ordering)
5. [Series Selection](#5-series-selection)
6. [Intensity Normalization](#6-intensity-normalization)
7. [Laterality Handling](#7-laterality-handling)
8. [Physical Sampling](#8-physical-sampling)
9. [2.5D Construction](#9-25d-construction)
10. [Resizing](#10-resizing)
11. [Dataset API](#11-dataset-api)
12. [Caching](#12-caching)
13. [Quality Control](#13-quality-control)
14. [Known Limitations](#14-known-limitations)
15. [Configurable Parameters](#15-configurable-parameters)
16. [Future Experiments](#16-future-experiments)

---

## 1. Pipeline Overview

### OBSERVED FACT
The RSNA knee MRI corpus consists of 4,407 studies with 24,371 series. The on-disk subset contains 30 series (5 studies, 1,067 DICOM files). Series are organized by anatomical plane: Sagittal (9,864), Coronal (8,609), Axial (5,898). Fluid-sensitive and fat-suppressed sequences are perfectly paired in this corpus.

### ENGINEERING DECISION
The pipeline converts heterogeneous knee MRI DICOM data into anatomically ordered, normalized 2.5D image samples. Every transformation is deterministic, reversible (metadata preserved), and configurable. Raw DICOM files are never modified.

### HIGH-LEVEL FLOW
```
DICOM → Series Discovery → DICOM Decoding → Physical Geometry Reconstruction
    → Anatomical Slice Ordering → Series Characterization → Candidate Series Ranking
    → Series Selection → Intensity Normalization → Physical/Anatomical Sampling
    → 2.5D Triplet Construction → Image Resizing / Tensor Conversion
    → Model-Ready Samples + Full Provenance Metadata
```

### KEY PRINCIPLES
- **No silent data loss**: Every skipped series/sample is logged with reason
- **Full provenance**: Every 2.5D sample traces back to original DICOM slices
- **Deterministic**: Same config + same input = identical output (when augmentation disabled)
- **Configurable**: All hyperparameters externalized to `configs/preprocessing.yaml`
- **Modular**: Each stage independently testable and replaceable

---

## 2. DICOM Loading

### OBSERVED FACT
- Transfer syntax: ExplicitVRLittleEndian (100% on subset)
- PhotometricInterpretation: MONOCHROME2 (100%)
- BitsAllocated=16, BitsStored=12, HighBit=11, PixelRepresentation=0
- Modality=MR (100%)
- 12-bit MRI stored in uint16 containers

### ENGINEERING DECISION
Use `pydicom` for header and pixel decoding. The Stage-1 `orthovision.data.discovery` module handles recursive discovery, header extraction, and UID-based grouping. The `orthovision.data.pixels` module provides three array views:
- `raw`: pixel_array as stored
- `rescaled`: `raw * RescaleSlope + RescaleIntercept` (float64)
- `display_preview`: min-max [0,1] for visualization only

### CONFIGURATION
No DICOM-specific config; handled by Stage-1 discovery config (`configs/data.yaml`).

---

## 3. Geometry Reconstruction

### OBSERVED FACT (Stage 1)
- ImageOrientationPatient (IOP) present on 100% of on-disk slices
- ImagePositionPatient (IPP) present on 100%
- PixelSpacing present on 100%
- SliceThickness present on 100%
- SpacingBetweenSlices present on 85% (missing on 160-slice VIBE series)
- IOP constant within series (spread < 1e-6)
- Slice-normal sign varies freely across series (6 sign combinations observed)

### ENGINEERING DECISION
Slice normal **n = normalize(IOP_row × IOP_col)** in LPS frame.  
Plane classification: X-dominant → sagittal, Y-dominant → coronal, Z-dominant → axial.  
Thresholds: max|axis| ≥ 0.8 with margin ≥ 0.2 (from `configs/data.yaml`).

CSV `Anatomical_Plane` agrees with DICOM-derived plane 30/30 on subset. The pipeline:
- Derives plane from geometry (authoritative)
- Compares against CSV plane
- Emits WARNING on disagreement; never silently overwrites

### IMPLEMENTATION
`orthovision.preprocessing.geometry.build_geometry()` computes:
- `ordering_method`: "spatial" | "instance_number" | "sop_instance_uid"
- `normal`: slice normal in LPS (tuple of 3 floats) or None
- `positions_mm`: physical position along normal for each slice
- `median_gap_mm`, `gap_cv`: spacing statistics
- `duplicate_positions`: count of near-duplicate positions
- `span_mm`: total anatomical coverage
- `laterality`: majority Laterality tag (if present)
- `patient_position`: majority PatientPosition tag

---

## 4. Slice Ordering

### OBSERVED FACT (Stage 1)
- InstanceNumber is present, unique, and contiguous-from-1 in every series
- Spearman ρ(InstanceNumber, IPP·n) = ±1.000 in all 30 series
- Direction flips: 18 ascending / 12 descending (60% toward increasing patient position)
- Physical spacing from positions is essentially uniform (gap CV ≤ 1e-4)

### ENGINEERING DECISION
**Canonical order**: Sort by `IPP · n` ascending in patient LPS coordinates.  
This is deterministic across studies and independent of scanner instance-numbering habits.

**Fallback chain** (recorded in `OrderingAudit`):
1. Spatial: `IPP · n` (requires IPP + IOP on all slices, consistent normals)
2. InstanceNumber: ascending, then SOPInstanceUID tie-break
3. SOPInstanceUID: ascending

**Laterality convention**: Position-ascending alone doesn't encode left/right. The pipeline records `laterality` from DICOM tags but does **not** flip images. Laterality is preserved as metadata for downstream use.

### REPRODUCIBILITY
Tests in `tests/test_preprocessing_geometry.py` verify:
- Deterministic ordering across runs
- Physical ordering matches IPP·n
- Fallback chain behavior with synthetic missing-geometry data
- Series with reversed InstanceNumber correctly ordered anatomically

---

## 5. Series Selection

### OBSERVED FACT
- Mean 5.53 series per study (median 5, max 14)
- Multiple series can share the same anatomical plane
- Fluid_Sensitive ≡ Fat_Suppression (perfectly paired in corpus)

### ENGINEERING DECISION
Two-stage selection:
1. **Hard filter**: Keep only series matching requested plane (CSV or derived)
2. **Soft ranking**: Score remaining candidates via weighted components in [0, 1]

### RANKING COMPONENTS (all in [0, 1])
| Component | Formula | Rationale |
|-----------|---------|-----------|
| `fluid` | 1.0 if matches plane's fluid_preference else 0.0 | Clinical relevance |
| `slices` | `saturating((n_slices - min_usable) / 45)` | More slices = better sampling |
| `resolution` | `saturating(area / 512²)` | Higher resolution = more detail |
| `completeness` | Fraction of key metadata tags present | Data quality |
| `consistency` | 1.0 for spatial+plane_agree, penalties for gaps/duplicates/mismatch | Geometric reliability |

### DEFAULT WEIGHTS (configurable)
```yaml
ranking_weights:
  fluid: 1.5
  slices: 1.0
  resolution: 0.5
  completeness: 1.0
  consistency: 2.0
```

### TIE-BREAK
Deterministic: score descending → slice count descending → SeriesInstanceUID ascending.

### OUTPUT
`SelectionReport` per plane with:
- `selected_series_uid`
- `candidates`: list of `CandidateScore` with `score`, `reasons`, `components`
- `notes`

### CONFIGURATION
All weights, fluid preferences, and planes in `configs/preprocessing.yaml`. The ranking is an **ENGINEERING BASELINE**, not a medically validated optimum.

---

## 6. Intensity Normalization

### OBSERVED FACT (Stage 1)
- Raw arrays: uint16, global range [0, 2179] (12-bit headroom unused)
- After rescale: global max ≈ 4,241
- RescaleSlope non-trivial (≈2.41–3.30) on 94/1,067 files; constant within series but **partial-tagging** in 20 series
- Fluid-sensitive series systematically darker than non-fluid-sensitive
- Sequences NOT intensity-standardized across series/studies
- No HU-like semantics; all grayscale MONOCHROME2

### ENGINEERING DECISION
Pipeline per slice (or triplet):
1. **Rescale** (already applied by `orthovision.data.pixels`): `raw * slope + intercept`
2. **MONOCHROME1 inversion**: `hi - (arr - lo)` if photometric=MONOCHROME1
3. **Robust clipping**: Percentile-based (default 0.5th / 99.5th)
4. **Normalization**: Two methods supported

### NORMALIZATION METHODS
| Method | Scope | Output Range | Use Case |
|--------|-------|--------------|----------|
| `percentile` | slice | [0, 1] | Default; bounded, suits patch embeddings |
| `percentile` | triplet | [0, 1] | Preserves relative intensity across channels |
| `zscore` | slice | (-∞, +∞) | Experiments; unbounded, fill value critical |
| `zscore` | triplet | (-∞, +∞) | Preserves relative intensity, shared moments |

### DEFAULT
```yaml
normalization:
  method: percentile
  scope: slice
  clip_percentiles: [0.5, 99.5]
```

### WHY NOT CT HOUNSFIELD
MRI intensities are arbitrary; no physical calibration. Percentile normalization to [0, 1] keeps letterbox padding (value=0) semantically meaningful as "black/background".

---

## 7. Laterality Handling

### OBSERVED FACT
- Laterality tag present in some series (L/R)
- PatientPosition tag present (HFS/FFS etc.)
- Slice-normal sign varies freely; position-ascending order doesn't indicate left vs right

### ENGINEERING DECISION
- **Do NOT blindly horizontally flip images**
- **Do NOT modify orientation** unless clear documented reason
- Preserve original orientation metadata in every sample
- Record:
  - `original_orientation`: IOP from DICOM
  - `detected_plane`: sagittal/coronal/axial from geometry
  - `laterality`: majority Laterality tag (or "unknown")
  - `patient_position`: majority PatientPosition tag
  - `any_transformation_applied`: always "none" currently

If laterality cannot be reliably determined → record as `"unknown"`.

### PROVENANCE
Every sample metadata includes:
```json
{
  "plane_derived": "sagittal",
  "laterality": "L",
  "patient_position": "HFS",
  "ordering_method": "spatial",
  "slice_normal_lps": [0.0, 1.0, 0.0],
  "channel_roles": ["previous", "center", "next"]
}
```

---

## 8. Physical Sampling

### ENGINEERING DECISION
Sample **anatomical locations**, not arbitrary slice indices. The sampler uses the physically ordered slice positions.

### DEFAULT STRATEGY
- `num_samples_per_plane: 6`
- Even spacing in anatomical position space: fractions in `[margin, 1-margin]`
- Default `boundary_margin: 0.06` (avoids ~6% at each end)
- If `sample_positions` configured explicitly, use those fractions instead
- Deterministic: same config + same series = same indices

### CONFIGURATION
```yaml
sampling:
  num_samples_per_plane: 6
  sample_positions: null      # override with explicit fractions [0,1]
  boundary_margin: 0.06
```

### OUTPUT
For each sample: `TripletPlan` with
- `center_index`: index in ordered series
- `slice_indices`: (prev, center, next) indices
- `physical_positions_mm`: physical positions for each channel

---

## 9. 2.5D Construction

### ENGINEERING DECISION
Each sample = 3 neighboring slices → 3 channels:
- Channel 0: previous slice
- Channel 1: center slice  
- Channel 2: next slice

### NEIGHBOR STRATEGIES
| Strategy | Description | Config |
|----------|-------------|--------|
| `index` | Fixed index offsets (default: -1, 0, +1) | `neighbor_offsets` |
| `physical_distance` | Find slices closest to center ± target_spacing_mm | `neighbor_target_spacing_mm`, `neighbor_max_search_slices` |

**Current default**: `index` with offsets `(-1, 0, 1)`.  
**Extensible**: `physical_distance` strategy implemented for future use when slice spacing varies substantially.

### PROVENANCE PRESERVED
Every sample metadata includes:
```json
{
  "center_slice_index": 15,
  "slice_indices": [14, 15, 16],
  "physical_positions_mm": [-12.3, -9.0, -5.7],
  "channel_roles": ["previous", "center", "next"],
  "sop_instance_uids": ["...", "...", "..."],
  "source_paths": ["...", "...", "..."]
}
```

---

## 10. Resizing

### ENGINEERING DECISION
**Aspect-preserving letterbox resize** to target size (default 224×224):
1. Scale by `min(target_h/h, target_w/w)`
2. Resize with configurable interpolation (bilinear/nearest)
3. Center pad with configurable fill value (default 0.0 = black)

**No cropping** — aggressive cropping could remove anatomy (Stage-1 FACT: 640×540 non-square matrices exist).

### CONFIGURATION
```yaml
image_size: [224, 224]
resize:
  interpolation: bilinear
  fill: 0.0
```

### METADATA
Each channel gets `ResizeMeta`:
```json
{
  "original_shape": [384, 384],
  "resized_shape": [224, 224],
  "scale": 0.583,
  "pad_top": 0, "pad_bottom": 0, "pad_left": 47, "pad_right": 47,
  "interpolation": "bilinear",
  "fill": 0.0
}
```

---

## 11. Dataset API

### PyTorch-COMPATIBLE INTERFACE
`orthovision.preprocessing.dataset.KneeStudyDataset` implements `torch.utils.data.Dataset`.

### `__getitem__(index)` RETURNS
```python
{
    "images": FloatTensor[N, 3, H, W],           # N 2.5D samples for this study
    "sample_metadata": List[Dict],               # Length N, full provenance
    "labels": FloatTensor[12],                   # NaN where unlabeled (4,349/4,407 studies)
    "study_uid": str
}
```

### MODES
| Mode | Behavior |
|------|----------|
| `train` | Transform hook may run (disabled by default → deterministic) |
| `val` | Deterministic; labels used if present |
| `inference` | Identical outputs; labels NaN when absent |

### KEY PROPERTIES
- Unlabeled studies kept in index; labels = NaN (never imputed, never dropped)
- Missing fields represented explicitly (not silently dropped)
- Transform only applied in `train` mode when provided

---

## 12. Caching

### ENGINEERING DECISION
Config-keyed cache for expensive DICOM processing. **Not** a full-dataset precompute.

### CACHE KEY
`SHA256(pipeline_version + config_fingerprint + series_uid + ordered_SOP_UIDs)[:24]`

Any config change → new key → automatic invalidation.

### STORAGE
One `.npz` per series: `images` [N, 3, H, W] float32 + `metadata_json` string.

### CONFIGURATION
```yaml
cache:
  enabled: true
  dir: outputs/cache/preprocessing
```

### BEHAVIOR
- Enabled by default
- Corrupt cache entries logged and skipped (never crash)
- Manual invalidation helper: `cache.invalidate_series(cfg, series_uid, sop_uids)`

---

## 13. Quality Control

### VALIDATION CHECKS (per series)
| Check | Severity | Condition |
|-------|----------|-----------|
| empty_series | BLOCKING | 0 slices |
| insufficient_slices | BLOCKING | < min_slices_usable (default 3) |
| missing_geometry | WARNING | Fallback to InstanceNumber/SOP ordering |
| duplicate_positions | WARNING | Near-duplicate positions (eps=1mm) |
| inconsistent_spacing | WARNING | Gap CV > 0.15 |
| inconsistent_dimensions | BLOCKING | Mixed matrix sizes within series |
| plane_mismatch | WARNING | CSV plane ≠ DICOM-derived plane |
| unusual_orientation | WARNING | Plane could not be derived from IOP |
| unexpected_modality | WARNING | Modality ≠ MR |
| invalid_pixels | BLOCKING | Empty/non-finite/constant pixel array |
| unreadable_dicom | BLOCKING (strict) | Pixel decode failure |

### MODES
| Mode | Behavior |
|------|----------|
| `permissive` (default) | Log warnings, skip bad series/samples, continue |
| `strict` | Raise `SeriesQCError` on any BLOCKING issue |

### PER-SLICE CHECKS
- Empty array → BLOCKING
- Non-finite values → BLOCKING
- Constant-valued → WARNING

---

## 14. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Plane agreement only verified on 30/24,371 series | May not hold corpus-wide | Cross-check CSV vs derived at runtime; WARNING on mismatch |
| Header conventions (12-bit, partial slopes) only verified on subset | Full corpus may differ | `scan_dicom_headers.py` can validate on full data |
| Laterality tags not universal | Laterality often "unknown" | Preserve as metadata; don't assume |
| Fluid_Sensitive ≡ Fat_Suppression only on subset | May not hold full corpus | Treat as separate signals in config |
| 160-slice VIBE outlier vs typical 16–48 slice TSE | Slice-count logic must handle variance | Configurable `num_samples_per_plane`, `boundary_margin` |
| Non-square 640×540 matrices exist | Resize must not assume squareness | Letterbox preserves aspect |
| Partial RescaleSlope tagging within series | Naive per-file rescale creates jumps | Series-level rescale policy (apply series median slope) |
| Test set has no local DICOM | Local inference needs held-out train studies | Dataset supports arbitrary study lists |

---

## 15. Configurable Parameters

All parameters in `configs/preprocessing.yaml`:

```yaml
# Pipeline mode
mode: permissive                    # strict | permissive

# Planes to process
planes: [Sagittal, Coronal, Axial]

# Series selection
selection:
  fluid_preference:                 # 1, 0, or none per plane
    Sagittal: 1
    Coronal: 1
    Axial: none
  ranking_weights:                  # Soft ranking component weights
    fluid: 1.5
    slices: 1.0
    resolution: 0.5
    completeness: 1.0
    consistency: 2.0

# Intensity normalization
normalization:
  method: percentile                # percentile | zscore
  scope: slice                      # slice | triplet
  clip_percentiles: [0.5, 99.5]

# Resize
image_size: [224, 224]
resize:
  interpolation: bilinear           # bilinear | nearest
  fill: 0.0

# Sampling
sampling:
  num_samples_per_plane: 6
  sample_positions: null            # explicit fractions [0,1] or null
  boundary_margin: 0.06

# Neighbors
neighbors:
  strategy: index                   # index | physical_distance
  offsets: [-1, 0, 1]               # channel order: prev, center, next
  target_spacing_mm: null
  max_search_slices: 5

# Quality control
qc:
  min_slices_usable: 3
  duplicate_position_epsilon_mm: 1.0e-3
  unusual_spacing_cv: 0.15
  orientation_min_alignment: 0.8

# Cache
cache:
  enabled: true
  dir: outputs/cache/preprocessing
```

---

## 16. Future Experiments

| Experiment | Config Change | Hypothesis |
|------------|---------------|------------|
| Physical-distance neighbors | `neighbors.strategy: physical_distance` + `target_spacing_mm` | Better for variable spacing (VIBE vs TSE) |
| Triplet-scope normalization | `normalization.scope: triplet` | Preserves inter-channel contrast for pathology |
| Z-score normalization | `normalization.method: zscore` | May suit DINOv2 pretraining distribution |
| More/fewer samples | `sampling.num_samples_per_plane: 8` or `4` | Trade-off coverage vs compute |
| Custom anatomical locations | `sampling.sample_positions: [0.15, 0.3, 0.5, 0.7, 0.85]` | Target specific anatomies (femoral condyle, meniscus, etc.) |
| Different fluid preferences | `selection.fluid_preference.Axial: 1` | Test if axial FS adds value |
| Reweight ranking | Increase `consistency` weight | Prioritize geometrically clean series |
| Larger image size | `image_size: [384, 384]` | More detail for fine structures |
| Different interpolation | `resize.interpolation: nearest` | Faster, may preserve edges better |

---

## Appendix: File Structure

```
src/orthovision/preprocessing/
├── __init__.py              # Public exports
├── config.py                # PreprocessConfig, load_preprocess_config
├── discovery.py             # SeriesInfo, discover_studies
├── geometry.py              # SliceRow, SeriesGeometry, order_slices
├── pipeline.py              # StudyPreprocessor, StudyResult
├── selection.py             # SelectionReport, rank_candidates, select_study_series
├── intensity.py             # prepare_slice, prepare_triplet, NormalizationStats
├── sampling.py              # TripletPlan, sample_locations, plan_series_samples
├── resize.py                # letterbox_resize, stack_triplet_channels, ResizeMeta
├── qc.py                    # SeriesQC, validate_series, check_pixel_array
├── cache.py                 # SampleCache, series_cache_key
└── dataset.py               # KneeStudyDataset

configs/
├── preprocessing.yaml       # Stage-2 config
└── data.yaml                # Stage-1 config (paths, plane thresholds)

notebooks/
├── 01_dataset_eda.ipynb     # Stage-1 EDA
└── 02_2p5d_visualization.ipynb  # Stage-2 visual validation

tests/
├── test_preprocessing_geometry.py
├── test_preprocessing_intensity_sampling_resize.py
├── test_preprocessing_pipeline.py
├── test_preprocessing_selection.py
└── test_spatial.py          # Stage-1 spatial tests
```

---

## Definition of Done — Stage 2 Checklist

- [x] Anatomical series can be discovered
- [x] Slices are physically ordered
- [x] Series candidates can be ranked
- [x] A deterministic series-selection mechanism exists
- [x] MRI intensity normalization works
- [x] Laterality/orientation metadata is preserved
- [x] Configurable physical sampling works
- [x] 2.5D triplets are generated
- [x] Final image tensors are generated
- [x] Full provenance metadata is preserved
- [x] PyTorch Dataset interface exists
- [x] Caching exists
- [x] QC checks exist
- [x] Visualization notebook exists
- [x] Tests exist and pass (67/67)
- [x] PREPROCESSING_PIPELINE.md exists
- [x] No raw data was modified
- [x] No neural network was implemented

---

## Final Output Summary

### Files Created/Modified
1. **New**: `scripts/build_2p5d_visualization_notebook.py` — Notebook generator
2. **New**: `notebooks/02_2p5d_visualization.ipynb` — Visual validation notebook
3. **New**: `docs/PREPROCESSING_PIPELINE.md` — This documentation
4. **Fixed**: `tests/test_preprocessing_pipeline.py` — Corrected label index in test

### Tests Executed
- All 67 tests pass (41 preprocessing + 26 Stage-1)
- Coverage: geometry, intensity, sampling, resize, pipeline, selection, caching, dataset API, QC

### Example Study Processing (Synthetic)
- Study: `1.2.826.555.1` (synthetic, 3 planes)
- Planes extracted: **3** (Sagittal, Coronal, Axial)
- 2.5D samples produced: **12** (4 per plane with test config)
- All series processed successfully (no skips in synthetic test)

### Important Preprocessing Decisions
1. **Physical ordering over InstanceNumber** — Canonical LPS ascending
2. **Percentile [0,1] normalization** — Bounded, padding-semantic, configurable
3. **No image flipping** — Laterality preserved as metadata only
4. **Letterbox resize** — No cropping, aspect preserved
5. **Config-keyed caching** — Automatic invalidation on config change
6. **Permissive default mode** — Log and skip, don't crash

### Remaining Uncertainties
1. Does 100% plane agreement hold on full 24,371 series?
2. Do partial RescaleSlope conventions hold corpus-wide?
3. Can multiple studies share PatientID (leakage risk)?
4. Optimal intensity normalization per sequence family (pd_tse_fs, t1_vibe, t2_tse)?
5. Does Fluid_Sensitive ≡ Fat_Suppression hold corpus-wide?

### Recommended Experiments for Stage 3
1. **Neighbor strategy**: Switch to `physical_distance` for variable-spacing series
2. **Normalization scope**: Try `triplet` to preserve inter-channel contrast
3. **Sampling density**: Vary `num_samples_per_plane` (4, 6, 8, 10)
4. **Ranking weights**: Grid search over `ranking_weights` with downstream AUC
5. **Image size**: Test 224, 384, 512 for ViT/DINOv2 backbones
6. **Sequence-specific norms**: Cluster by SeriesDescription, learn per-cluster params

---

*End of documentation. Stage 2 complete. Awaiting Stage 3 instructions.*