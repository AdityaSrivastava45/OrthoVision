# OrthoVision

Explainable multimodal AI for **knee MRI** analysis.

**Research question:** Can radiology reports be used as privileged clinical information *during training* to improve an MRI-only deep learning model for multi-abnormality knee MRI classification?

**Deployment constraint:** reports are available at train time only. Final inference is **MRI-only**.

This repository currently implements **milestone 1 only**: DICOM dataset understanding (discovery, series grouping, slice ordering, plane identification, validation, manifest, visualization). No neural networks, APIs, or explainability.

---

## Assessment of the starting tree

| Item | Status |
|------|--------|
| Existing code / notebooks / DICOM utilities | None (empty project directory) |
| Language / packaging | Python 3.10+, `pyproject.toml` + `pip` |
| Local knee MRI files in-repo | **Not present** (`data/raw/` is a placeholder) |

Nothing was overwritten. The data layer is a new package under `src/orthovision`.

---

## OBSERVED DATASET FACTS vs ENGINEERING ASSUMPTIONS

### Observed (this environment)

- The OrthoVision project folder contained no source files and no DICOM objects at first inspection.
- `data/raw` is an empty placeholder until you point `configs/data.yaml` at a real archive.
- No prevalence, sequence names, slice counts, or demographics are claimed here because they were not measured on a real corpus.

### Engineering assumptions (not dataset facts)

- Instances are grouped by `StudyInstanceUID` / `SeriesInstanceUID`, not by folder names.
- Patient coordinate system is DICOM **LPS**. Slice normal = row direction × column direction from `ImageOrientationPatient`.
- Approximate plane: dominant |normal| along X → sagittal, Y → coronal, Z → axial. If alignment is weak or two axes are close, plane is **unknown** (oblique), not guessed from `SeriesDescription`.
- `SeriesDescription` is stored and parsed only as a **hint**. Spatial geometry always wins.
- Slice order uses projected `ImagePositionPatient` when every instance has IPP+IOP; otherwise **fallback** to `InstanceNumber`, then `SOPInstanceUID`. Fallbacks are audited.
- `RescaleSlope` / `RescaleIntercept` are applied only in the **rescaled** array. VOI windowing is optional and **display-only**.
- Multi-frame Enhanced MR is detected and flagged; frames are **not** expanded in this milestone.
- Compressed transfer syntaxes may require `pip install 'orthovision[jpeg]'` (pylibjpeg). Decode failures are recorded, not dropped.

---

## Architecture

```
dataset_root (config)
    → recursive file scan
    → DICOM headers (stop_before_pixels)
    → Study (StudyInstanceUID)
        → Series (SeriesInstanceUID)
            → ordered SliceRecord
            → plane audit + validation
    → series_manifest.csv / scan_summary.json
    → matplotlib mosaics
```

ML code should call `KneeMRIDataset.get_study(study_id)` and use UIDs + arrays from `orthovision.data.pixels`. Do not scrape the filesystem in training code.

### Package layout

| Path | Role |
|------|------|
| `configs/data.yaml` | dataset root, outputs, workers, thresholds |
| `src/orthovision/data/discovery.py` | recursive DICOM detection |
| `src/orthovision/data/spatial.py` | ordering + plane |
| `src/orthovision/data/pixels.py` | raw / rescaled / display preview |
| `src/orthovision/data/validate.py` | VALID / WARNING / INVALID |
| `src/orthovision/data/catalog.py` | `KneeMRIDataset` |
| `src/orthovision/data/manifest.py` | CSV/JSON metadata (no pixels) |
| `src/orthovision/data/visualize.py` | slice mosaics |
| `notebooks/inspect_study.ipynb` | manual order/plane QA |
| `tests/` | synthetic DICOM tests (no real PHI) |

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional JPEG/RLE decoders:

```bash
pip install -e ".[jpeg]"
```

Edit `configs/data.yaml`:

```yaml
dataset_root: data/raw   # or an absolute path to the MRI archive
output_dir: outputs
num_workers: 4
logging_level: INFO
```

---

## Commands

Scan and print counts:

```bash
python -m orthovision.cli --config configs/data.yaml inspect --manifest
```

Study summary (replace UID):

```bash
python -m orthovision.cli study-summary --study STUDY_INSTANCE_UID
```

Visualize ordered slices:

```bash
python -m orthovision.cli visualize --study STUDY_INSTANCE_UID --n-slices 7
```

Tests:

```bash
pytest
```

Python API:

```python
from orthovision import KneeMRIDataset

ds = KneeMRIDataset()          # loads configs/data.yaml
ds.scan()
study = ds.get_study(ds.study_ids()[0])
series = study.series[0]
print(series.plane, series.ordering_audit.method, series.validation.status)

from orthovision.data.pixels import decode_path, display_preview
payload = decode_path(series.slices[0].source_path)
# payload.raw, payload.rescaled  — training normalization is a later milestone
```

---

## Slice ordering (audit)

1. If all instances have `ImageOrientationPatient` and `ImagePositionPatient`, compute one slice normal and sort by `IPP · n`.
2. Else sort by `InstanceNumber` (recorded as fallback).
3. Else sort by `SOPInstanceUID`.

`OrderingAudit` on each series stores method, normal, positions, and notes.

## Plane detection

See docstring in `src/orthovision/data/spatial.py`. Ambiguous geometry → `unknown`.

## Validation statuses

- **VALID** — inspected checks passed.
- **WARNING** — usable but issues (missing optional tags, fallback order, few slices, unknown plane, spacing CV, description/geometry disagreement, multi-frame, mixed transfer syntax).
- **INVALID** — empty, unreadable pixels/headers in-series, inconsistent matrix size, inconsistent orientation.

Reasons are always listed in `validation.messages` and in the manifest.

## Manifest

Written under `outputs/manifests/` (configurable):

- `series_manifest.csv` / `.json` — one row per series
- `file_issues.csv` — unreadable / missing-UID files
- `scan_summary.json` — counts + study summaries

Pixel arrays are never stored in the manifest.

## Visualization

Mosaics sample evenly along the **already ordered** volume so you can judge whether anatomy progresses consistently. This is research QA, not a viewer product.

**Manual inspection is still required** before trusting order/plane on a real archive: tests use synthetic geometry, not clinical MRI.

## Known limitations

- No 2.5D sampling, resampling, or intensity model-prep yet (next milestone).
- Multi-frame series are not exploded into per-frame slices.
- Compressed syntaxes need extra packages; unsupported syntaxes fail loudly.
- `data/raw` in this clone has no real studies; run `inspect` after configuring the archive.
- DICOM files without a 128-byte preamble are accepted only if a forced read yields SOP Class/Instance UIDs.
- Plane thresholds (`plane_min_alignment`, `plane_min_axis_separation`) are engineering defaults.

## Next milestone (out of scope here)

2.5D MRI preprocessing and sampling on top of `Study` / `Series` / ordered slices — still MRI-only at test time; reports remain train-time privileged data.
