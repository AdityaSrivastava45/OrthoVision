"""Series discovery: header table + train_series.csv -> SeriesInfo objects.

Grouping is by DICOM UIDs only (Stage-1 FACT: folder names are series UIDs in
this layout, and CSV/UID agreement was verified 100 % on the subset).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from orthovision.preprocessing.config import PreprocessConfig
from orthovision.preprocessing.geometry import SeriesGeometry, SliceRow, order_slices

AXES = ("sagittal", "coronal", "axial")
CSV_TO_DERIVED = {"Sagittal": "sagittal", "Coronal": "coronal", "Axial": "axial"}


def _parse_vec(text):
    if not isinstance(text, str):
        return None
    try:
        return tuple(float(x) for x in text.split(";"))
    except ValueError:
        return None


@dataclass
class SeriesInfo:
    study_uid: str
    series_uid: str
    plane_csv: str | None
    plane_derived: str | None
    fluid_sensitive: int | None
    fat_suppression: int | None
    n_slices: int
    rows: int | None
    columns: int | None
    pixel_spacing_mm: tuple[float, float] | None
    slice_thickness_mm: float | None
    slice_spacing_mm: float | None
    orientation_normal: tuple[float, float, float] | None
    ordering_method: str
    laterality: str | None
    patient_position: str | None
    series_description: str | None
    modality: str | None
    transfer_syntax_uid: str | None
    completeness: float
    slices_ordered: list[SliceRow]
    geometry: SeriesGeometry
    source_paths: list[str] = field(default_factory=list)

    def plane_agrees(self) -> bool | None:
        if self.plane_derived is None or self.plane_csv is None:
            return None
        return CSV_TO_DERIVED.get(self.plane_csv) == self.plane_derived


def load_header_table(headers_csv: Path | str) -> pd.DataFrame:
    return pd.read_csv(
        headers_csv,
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )


def attach_series_attributes(hdr: pd.DataFrame, series_csv_path: Path | str) -> pd.DataFrame:
    ts = pd.read_csv(series_csv_path, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    attrs = ts.set_index("SeriesInstanceUID")[["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]]
    return hdr.join(attrs, on="SeriesInstanceUID")


def _row_to_slice_row(r) -> SliceRow:
    iop = _parse_vec(getattr(r, "ImageOrientationPatient", None))
    ipp = _parse_vec(getattr(r, "ImagePositionPatient", None))
    spacing_raw = getattr(r, "PixelSpacing", None)
    spacing = _parse_vec(spacing_raw) if isinstance(spacing_raw, str) else None
    pixel_spacing = (spacing[0], spacing[1]) if spacing and len(spacing) >= 2 else None
    thickness = getattr(r, "SliceThickness", None)

    def _float(value):
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    extra = {
        "laterality": getattr(r, "Laterality", None),
        "patient_position": getattr(r, "PatientPosition", None),
        "slice_thickness": _float(thickness),
    }
    inst = getattr(r, "InstanceNumber", None)
    return SliceRow(
        sop_uid=str(getattr(r, "SOPInstanceUID")),
        path=str(getattr(r, "path")),
        instance_number=int(inst) if inst is not None and not pd.isna(inst) else None,
        iop=iop if iop and len(iop) >= 6 else None,
        ipp=ipp if ipp and len(ipp) == 3 else None,
        rows=_int_or_none(getattr(r, "Rows", None)),
        columns=_int_or_none(getattr(r, "Columns", None)),
        pixel_spacing=pixel_spacing,
        rescale_slope=_float(getattr(r, "RescaleSlope", None)),
        photometric=getattr(r, "PhotometricInterpretation", None),
        extra=extra,
    )


COMPLETENESS_TAGS = (
    "rows",
    "columns",
    "pixel_spacing_mm",
    "slice_thickness_mm",
    "orientation_normal",
    "laterality",
)


def build_series_info(study_uid: str, series_uid: str, grp: pd.DataFrame) -> SeriesInfo:
    slices = [_row_to_slice_row(r) for r in grp.itertuples(index=False)]
    ordered, geometry = order_slices(slices)
    first = ordered[0]

    present = sum(
        1
        for name in COMPLETENESS_TAGS
        if getattr(first, name, None) is not None or (
            name == "orientation_normal" and geometry.normal is not None
        )
    )
    completeness = present / len(COMPLETENESS_TAGS)

    derived_plane = None
    if geometry.normal is not None:
        abs_n = np.abs(geometry.normal)
        best = int(np.argmax(abs_n))
        second = float(np.sort(abs_n)[-2])
        if abs_n[best] >= 0.8 and (abs_n[best] - second) >= 0.2:
            derived_plane = AXES[best]

    gaps_ok_spacing = geometry.median_gap_mm if geometry.ordering_method == "spatial" else None
    thickness = first.extra.get("slice_thickness")
    return SeriesInfo(
        study_uid=study_uid,
        series_uid=series_uid,
        plane_csv=getattr(grp.iloc[0], "Anatomical_Plane", None) if hasattr(grp.iloc[0], "Anatomical_Plane") else None,
        plane_derived=derived_plane,
        fluid_sensitive=_int_or_none(getattr(grp.iloc[0], "Fluid_Sensitive", None)),
        fat_suppression=_int_or_none(getattr(grp.iloc[0], "Fat_Suppression", None)),
        n_slices=len(ordered),
        rows=first.rows,
        columns=first.columns,
        pixel_spacing_mm=first.pixel_spacing,
        slice_thickness_mm=thickness,
        slice_spacing_mm=gaps_ok_spacing,
        orientation_normal=geometry.normal,
        ordering_method=geometry.ordering_method,
        laterality=geometry.laterality,
        patient_position=geometry.patient_position,
        series_description=getattr(grp.iloc[0], "SeriesDescription", None),
        modality=getattr(grp.iloc[0], "Modality", None),
        transfer_syntax_uid=getattr(grp.iloc[0], "TransferSyntaxUID", None),
        completeness=completeness,
        slices_ordered=ordered,
        geometry=geometry,
        source_paths=[s.path for s in ordered],
    )


def _int_or_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def discover_studies(
    headers_csv: Path | str,
    series_csv_path: Path | str,
    cfg: PreprocessConfig | None = None,
) -> dict[str, list[SeriesInfo]]:
    """Return study_uid -> [SeriesInfo], deterministic order everywhere."""
    del cfg
    hdr = attach_series_attributes(load_header_table(headers_csv), series_csv_path)
    studies: dict[str, list[SeriesInfo]] = {}
    for (study_uid, series_uid), grp in hdr.groupby(["StudyInstanceUID", "SeriesInstanceUID"], sort=True):
        info = build_series_info(str(study_uid), str(series_uid), grp)
        studies.setdefault(str(study_uid), []).append(info)
    for lst in studies.values():
        lst.sort(key=lambda s: s.series_uid)
    return dict(sorted(studies.items()))


def ensure_header_cache(
    data_root: Path | str,
    cache_csv: Path | str,
    *,
    num_workers: int = 4,
) -> Path:
    """Run the Stage-1 header scan when the cache file is absent."""
    from orthovision.data.discovery import discover_dicom_headers

    cache_csv = Path(cache_csv)
    if cache_csv.exists():
        return cache_csv
    root = Path(data_root)
    headers, issues, counts = discover_dicom_headers(root, num_workers=num_workers)
    rows = []
    fields = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "SeriesDescription",
        "Modality",
        "PhotometricInterpretation",
        "TransferSyntaxUID",
        "Rows",
        "Columns",
        "InstanceNumber",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "PixelSpacing",
        "SliceThickness",
        "SpacingBetweenSlices",
        "RescaleSlope",
        "RescaleIntercept",
        "Laterality",
        "PatientPosition",
    ]
    for h in headers:
        row = {"path": str(h.path.relative_to(root))}
        for f in fields:
            value = h.metadata.get(f)
            if isinstance(value, tuple):
                value = ";".join(f"{v:.8g}" for v in value)
            row[f] = value
        rows.append(row)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cache_csv, index=False)
    print(f"[discovery] scanned {counts['files_seen']} files -> {cache_csv}")
    return cache_csv


def _script_hint() -> str:
    return sys.argv[0] if sys.argv else ""
