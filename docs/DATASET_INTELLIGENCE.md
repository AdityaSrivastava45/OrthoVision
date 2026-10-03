# DATASET INTELLIGENCE — OrthoVision Stage 1

**Corpus:** RSNA knee MRI, 12-abnormality classification (ACL, MCL, Medial Meniscus,
Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's,
Contusion, Fracture).
**Analysis date:** 2026-08-26.
**Environment:** Python 3.14.6 · pydicom 3.0.2 · pandas 3.0.3 · numpy 2.5.0 ·
matplotlib 3.11.1 · Windows 11, single volume `C:`.

Every statement below is tagged:

- **FACT** — directly observed from the provided files in this run.
- **ASSUMPTION** — believed but not verified on this data.
- **RECOMMENDATION** — engineering decision derived from observations.

---

## 1. Dataset overview

- **FACT** `train.csv`: 4,407 rows = 4,407 unique `StudyInstanceUID`, columns
  `StudyInstanceUID, Report, <12 labels>`.
- **FACT** Only **58 of 4,407 studies (1.32 %)** carry any structured label;
  the other **4,349 have a free-text report and all-labels-NaN**.
- **FACT** `train_series.csv`: 24,371 rows, all `SeriesInstanceUID` unique,
  covering exactly the same 4,407 studies; columns
  `StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane`.
- **FACT** Local test split: `test.csv` = **3 studies**, `test_series.csv` = **15 series**,
  zero overlap with train (study or series level). No `test_series/` DICOM folder was provided.
- **FACT** `sample_submission.csv` matches test study IDs exactly; all values are 0.5
  → evaluation is per-label probability (AUC-style), 12 columns in fixed order.
- **FACT** Reports are multilingual (Spanish/German/English markers detected in
  681 / 259 / 1,666 reports respectively, categories overlap), median length 977 chars,
  range 52–4,743. File is valid UTF-8 (apparent mojibake is console-only).
- **FACT** On-disk DICOM subset (`data/raw/train_series/`): **30 top-level folders =
  30 SeriesInstanceUIDs belonging to only 5 studies** (the first five rows of
  `train.csv`), 1,067 DICOM objects, 0 unreadable.
- **FACT** On-disk layout is **flat**: `train_series/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`
  — NOT `<StudyInstanceUID>/<SeriesInstanceUID>/…` as originally described. Folder names
  equal header `SeriesInstanceUID` for 100 % of files; CSV series→study mapping agrees
  with header `StudyInstanceUID` for 100 % of files.

## 2. Repository structure

| Path | Role | Status |
|---|---|---|
| `src/orthovision/data/discovery.py` | recursive DICOM detection + header fields | pre-existing, extended additively (Bits*, Rescale*, SamplesPerPixel, PixelRepresentation) |
| `src/orthovision/data/spatial.py` | slice ordering + plane from geometry | pre-existing |
| `src/orthovision/data/pixels.py` | raw / rescaled / display decode | pre-existing |
| `src/orthovision/data/catalog.py` | Study/Series catalog | pre-existing |
| `src/orthovision/data/validate.py`, `manifest.py`, `visualize.py`, `tags.py`, `models.py`, `config.py` | support layers | pre-existing |
| `src/orthovision/data/tabular.py` | **NEW** CSV layer: schema-checked loaders, label stats, co-occurrence | this stage |
| `scripts/scan_dicom_headers.py` | full header scan → `outputs/reports/dicom_headers.csv` | this stage |
| `scripts/dataset_structure.py` | Phase 2 report | this stage |
| `scripts/label_analysis.py` | Phase 3 report | this stage |
| `scripts/dicom_analysis.py` | Phase 4 report (+90-slice pixel sample) | this stage |
| `scripts/orientation_analysis.py` | Phase 5 report | this stage |
| `scripts/slice_ordering_analysis.py` | Phase 6 report | this stage |
| `scripts/quality_audit.py` | Phase 7 full-pixel audit | this stage |
| `scripts/build_eda_notebook.py` | regenerates `notebooks/01_dataset_eda.ipynb` | this stage |
| `notebooks/01_dataset_eda.ipynb` | executed EDA with figures | this stage |
| `tests/test_tabular.py` | **NEW** tests for CSV layer | this stage |

Mapping to the suggested layout: `dicom_reader.py` ≡ existing `discovery.py`+`pixels.py`;
`metadata.py` ≡ existing `tags.py`+`spatial.py`; `dataset.py` ≡ existing `catalog.py`.
No duplicates were created.

## 3. Dataset statistics

**Full corpus scope (CSV):**

- **FACT** Series per study: mean **5.53**, median 5, min 3, max 14; mode 5 series (2,299 studies).
- **FACT** Plane distribution (series level): Sagittal **9,864** / Coronal **8,609** / Axial **5,898**.
- **FACT** `Fluid_Sensitive`=1 on **14,010** series, 0 on 10,361. `Fat_Suppression` has an
  **identical** marginal distribution, and the joint table contains only the two diagonal
  combos `(FS=0,Fat=0)` and `(FS=1,Fat=1)` per plane → the two columns are perfectly
  redundant in this corpus.
- **FACT** Per plane (full corpus): Axial 4,719 FS vs 1,179 non-FS; Coronal 4,624 vs 3,985;
  Sagittal 4,667 vs 5,197.

**On-disk subset scope (headers, n=1,067 files / 30 series / 5 studies):**

- **FACT** Slices per series: median **32**, min 16, max **160** (a 3D `t1_vibe_we_tra_cartilage`
  acquisition). Distribution: {16:3, 18:1, 22:2, 29:3, 30:4, 32:3, 34:2, 35:2, 38:6, 40:2, 48:1, 160:1}.
- **FACT** Every on-disk study's series set exactly equals its CSV series list
  (5/5/5/10/5); no partial studies, no orphans in either direction.
- **FACT** Matrix sizes observed: 384² (275), 640×640 (257→197 square + **60 at 640×540**),
  416² (200), 256² (160), 768² (68), plus smaller counts at 320², 560², 512², 576², 540-col variant.

## 4. Label statistics (58 labeled studies)

**FACT** All labels are binary {0.0, 1.0}; no partial/hard multi-values; no missing values
*within* labeled studies.

| Label | Pos | Neg | Prevalence |
|---|---|---|---|
| Effusion | 35 | 23 | **0.603** (most common) |
| Synovitis | 27 | 31 | 0.466 |
| Medial Meniscus | 26 | 32 | 0.448 |
| ACL | 24 | 34 | 0.414 |
| Lateral Meniscus | 23 | 35 | 0.397 |
| PF OA | 21 | 37 | 0.362 |
| Contusion | 19 | 39 | 0.328 |
| Fracture | 18 | 40 | 0.310 |
| Medial OA | 15 | 43 | 0.259 |
| Baker's | 12 | 46 | 0.207 |
| Lateral OA | 11 | 47 | 0.190 |
| MCL | 9 | 49 | **0.155** (rarest) |

- **FACT** Positives per labeled study: 1–9 (mode 3). No all-negative or all-positive study.
- **FACT** Strongest co-occurrence pairs (Jaccard among positives): Effusion–Synovitis
  **0.55** (joint n=22/58), MedialMeniscus–Effusion 0.45, ACL–Effusion 0.44, ACL–Contusion
  0.43, MedialOA–Baker's 0.42.
- **FACT** Imbalance evidence: MCL has 9 positives in 58 studies; with 4,349 studies having
  *no* structured label at all, supervised training on hard labels alone is impossible at
  corpus scale. (Loss design deliberately NOT decided here.)

## 5. DICOM metadata findings (subset, n=1,067)

- **FACT** Modality `MR` 1,067/1,067; no other modality.
- **FACT** Transfer syntax: `ExplicitVRLittleEndian` (1.2.840.10008.1.2.1) 1,067/1,067 —
  **no compression anywhere in the subset**; JPEG extras not needed for this data.
- **FACT** PhotometricInterpretation `MONOCHROME2` always; **MONOCHROME1 does not occur**.
- **FACT** SamplesPerPixel=1, BitsAllocated=16, BitsStored=12, HighBit=11,
  PixelRepresentation=0 (unsigned) — classic 12-bit MRI stored in uint16.
- **FACT** Tag presence: `ImageOrientationPatient`, `ImagePositionPatient`,
  `InstanceNumber`, `PixelSpacing`, `SliceThickness`, `Rows/Columns` → **100 % present**.
  `SpacingBetweenSlices` present on 907/1,067 (85 %; absent exactly on the 160-file VIBE series).
- **FACT** `RescaleSlope` present on 648/1,067 (61 %): value 1.0 on 554 files, non-trivial
  (≈2.41–3.30) on 94 files; `RescaleIntercept` = 0.0 wherever present. Slope is constant
  within a series but **present on only some slices within 20 series** (partial-tagging trap).
- **FACT** `ProtocolName`, `StudyDate` absent everywhere (de-identification).
  `SeriesDescription` present everywhere; `PatientID` pseudonymized UUIDs; manufacturers:
  Siemens Healthineers/SIEMENS (two spellings), Philips Healthcare.
- **ASSUMPTION** The full corpus shares these header conventions; verified so far only on
  the 5-study subset.

## 6. Image characteristics (pixel sample: first/middle/last of every series, n=90)

- **FACT** Raw arrays are `uint16`; observed global raw range **[0, 2179]** (12-bit headroom
  unused above ~2^11 in practice); after rescale the global max reaches ≈**4,241**.
- **FACT** Rescale slope materially changes intensities where ≠1 (e.g., raw max 843 →
  2,154 on a slope=2.55507 file): **rescale is not a no-op** on some sequences.
- **FACT** Fluid-sensitive series are systematically darker than non-fluid-sensitive
  (sample means ≈74–82 vs ≈171–207; p99 ≈497–773 vs ≈1,506–1,623) — sequences are **not**
  intensity-standardized across series/studies.
- **FACT** All grayscale (single channel), MONOCHROME2 → high value = bright; CT-style
  windowing is irrelevant; no HU-like semantics.
- **RECOMMENDATION** Always apply `RescaleSlope/Intercept` when present (float64 path),
  treat missing as identity, and never mix rescaled and raw values within one series.

## 7. Orientation findings (all 30 on-disk series)

Method: slice normal **n = IOP_row × IOP_col** (LPS); X-dominant→sagittal,
Y-dominant→coronal, Z-dominant→axial; thresholds max|axis| ≥ 0.8 with margin ≥ 0.2
(`configs/data.yaml`).

- **FACT** CSV `Anatomical_Plane` vs DICOM-derived plane agreement: **30/30 = 100 %**
  (Sagittal 11/11, Coronal 11/11, Axial 8/8). Zero unknown/oblique series; zero confusions.
- **FACT** IOP is constant within every series (spread < 1e-6).
- **FACT** Slice-normal sign varies freely across series (six sign combinations observed)
  → normal direction carries no canonical meaning by itself.
- **RECOMMENDATION** Trust geometry as ground truth for plane; keep CSV plane as a cheap
  cross-check and raise a WARNING on disagreement rather than silently correcting either side.
- **ASSUMPTION** Agreement extrapolates to the remaining 4,377 studies (verified on 30 series).

## 8. Slice-ordering findings (all 30 on-disk series)

- **FACT** `InstanceNumber` is present, **unique**, and contiguous-from-1 in every series.
- **FACT** Spearman ρ(InstanceNumber, IPP·n) = **exactly ±1.000 in all 30 series** —
  InstanceNumber always gives a perfect anatomical rank order *within* a series…
- **FACT** …but the direction flips between series: **18 ascending / 12 descending**
  (increasing InstanceNumber runs toward increasing patient position in only 60 %).
- **FACT** Physical spacing from positions is essentially perfectly uniform
  (gap CV ≤ 1e-4 per series); median gap ranges 0.6–7.72 mm; span 36–176 mm; no duplicate
  positions (ε=1 µm) anywhere.
- **RECOMMENDATION (canonical order)** Sort slices by `IPP · (IOP_row × IOP_col)` ascending
  in patient LPS coordinates. This is deterministic across studies and independent of each
  scanner's instance-numbering habit. Use InstanceNumber only as fallback when IPP/IOP are
  missing, and record which method was used. For cross-study consistency (later 2.5D
  sampling), also fix the anatomical *direction convention* explicitly (e.g., always
  left→right for sagittal) since position-ascending alone doesn't encode laterality.
- **NOT done here:** the final 2.5D sampler — later stage.

## 9. Data-quality audit (full pass over all 1,067 files)

Severity summary: **0 CRITICAL · 3 WARNING · 8 INFO** (`outputs/reports/quality_audit.json`).

| Severity | Finding |
|---|---|
| INFO | All 1,067 files decode; matrix size constant within every series; modality/syntax/photometric uniform; PatientID↔study 1:1 in subset; SeriesDescription constant per series; slices-per-series 16–160 |
| WARNING | **Non-square matrices**: 60 images are 640×540 (one series) — resizing must not assume squareness |
| WARNING | **Partial RescaleSlope tagging**: 20 series have slope on some slices only → naive per-file rescale creates intra-series intensity jumps |
| WARNING | **SpacingBetweenSlices missing** on 160 files (one VIBE series); harmless because IPP-based spacing works |
| — | No duplicate SOPInstanceUIDs; no byte-identical duplicate images (SHA-256 of pixel block); no constant/all-zero images; no corrupted headers; no orphaned series/files; folder-name↔header UID mismatches: none |

## 10. Important edge cases

1. **FACT** Flat disk layout keyed by *Series* UID — any code assuming Study-level folders breaks.
2. **FACT** NaN labels ≠ negative labels: 4,349 studies are unlabeled, not healthy.
3. **FACT** `Fluid_Sensitive` ≡ `Fat_Suppression` (perfectly paired columns in this corpus).
4. **FACT** InstanceNumber direction flip (18↑/12↓) — see §8.
5. **FACT** Partial rescale tagging within series — see §9.
6. **FACT** 160-slice 0.6 mm VIBE outlier vs typical 16–48 slice TSE stacks — slice-count
   based logic must not assume homogeneity.
7. **FACT** Non-square 640×540 images.
8. **FACT** Test side locally is only 3 studies / 15 series with no DICOM folder — local
   end-to-end inference checks must reuse held-out train studies instead.
9. **FACT** Reports exist for ALL 4,407 studies but must be treated as train-time-only.
10. **FACT** `PatientID` exists only inside DICOM headers (pseudonymized), not in any CSV.

## 11. Recommended preprocessing decisions (engineering, not modeling)

- **R1** Group instances exclusively by header UIDs (never folder names); cache a
  series manifest once via `scripts/scan_dicom_headers.py`.
- **R2** Plane: derive from IOP normal (thresholds 0.8 / 0.2); compare against CSV;
  warn on mismatch; never silently overwrite.
- **R3** Order: `IPP · n` ascending; fallback chain InstanceNumber → SOPInstanceUID;
  persist chosen method + slice normal per series for auditability.
- **R4** Decode with pydicom; apply rescale when tagged; keep raw vs rescaled separate
  (`orthovision.data.pixels` already does this); display previews are min-max scaled and
  never used for training math.
- **R5** Treat intensity normalization as an open experimental variable (§12) — do not
  bake percentile/z-score choices into the data layer yet.
- **R6** Keep raw tree immutable; all derived artifacts under `outputs/`.

## 12. Open questions (must be validated experimentally later)

1. **Q1** Does the 100 % plane agreement hold across the full 24,371-series corpus?
   (Verified on 30 series only.)
2. **Q2** Do header conventions (12-bit unsigned, ExplicitVR LE, partial slopes) hold
   corpus-wide, especially for the 4,402 studies not yet on disk?
3. **Q3** Can multiple studies share one pseudonymized `PatientID` in the full corpus?
   (Determines whether splits must be patient-level to prevent leakage.)
4. **Q4** Correct handling of partially-tagged rescale series: rescale-all-with-default-1
   vs per-series policy? Needs intensity-distribution impact analysis.
5. **Q5** Optimal intensity normalization per sequence family (SeriesDescription clusters
   like `pd_tse_fs_*`, `t1_vibe_*`, `t2_tse_*`) for the future backbone.
6. **Q6** Label strategy for 4,349 report-only studies (out of scope for Stage 1; the
   58 hard labels alone cannot supervise a 12-head model at scale).
7. **Q7** Whether `Fluid_Sensitive≡Fat_Suppression` redundancy persists corpus-wide
   (affects how many sequence-slot features are real degrees of freedom).
8. **Q8** Laterality encoding: position-ascending order doesn't say left-vs-right knee;
   needs explicit convention before 2.5D sampling.

## Reproduction commands

```bash
# environment (system Python 3.14.6 already satisfies deps)
pip install -e ".[dev]"

python scripts/scan_dicom_headers.py --workers 4      # Phase 4 foundation (~seconds)
python scripts/dataset_structure.py                   # Phase 2 -> outputs/reports/*.json
python scripts/label_analysis.py                      # Phase 3
python scripts/dicom_analysis.py                      # Phase 4 pixels (stratified sample)
python scripts/orientation_analysis.py                # Phase 5
python scripts/slice_ordering_analysis.py             # Phase 6
python scripts/quality_audit.py                       # Phase 7 (full pixel pass)
python scripts/build_eda_notebook.py                  # regenerate clean notebook
python -m pytest tests/                               # 26 tests, all green
```

All reports land in `outputs/reports/`; raw data under `data/raw/` was never modified
(reads only; verified by hash-preserving workflow).
