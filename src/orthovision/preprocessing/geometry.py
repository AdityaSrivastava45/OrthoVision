"""Physical geometry reconstruction for one series.

Builds the canonical slice ordering from ImageOrientationPatient /
ImagePositionPatient (Stage-1 finding: 100 % coverage on the on-disk subset),
with a documented fallback chain. Deterministic by construction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from orthovision.data.spatial import project_position, slice_normal


ORDERING_SPATIAL = "spatial"
ORDERING_INSTANCE = "instance_number"
ORDERING_SOP = "sop_instance_uid"


@dataclass
class SliceRow:
    """Minimal per-instance view over the cached header table."""

    sop_uid: str
    path: str
    instance_number: int | None
    iop: tuple[float, ...] | None
    ipp: tuple[float, float, float] | None
    rows: int | None
    columns: int | None
    pixel_spacing: tuple[float, float] | None
    rescale_slope: float | None
    photometric: str | None
    extra: dict = field(default_factory=dict)


@dataclass
class SeriesGeometry:
    ordering_method: str
    normal: tuple[float, float, float] | None
    positions_mm: list[float | None]
    median_gap_mm: float | None
    gap_cv: float | None
    duplicate_positions: int
    span_mm: float | None
    laterality: str | None
    patient_position: str | None


def _majority_non_empty(values: list[str | None]) -> str | None:
    cleaned: list[str] = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        text = str(v).strip()
        if text:
            cleaned.append(text.upper())
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def build_geometry(slices: list[SliceRow], *, dup_epsilon_mm: float = 1e-3) -> SeriesGeometry:
    normals: list[np.ndarray] = []
    usable = True
    for s in slices:
        n = slice_normal(s.iop)
        if n is None or s.ipp is None:
            usable = False
            break
        normals.append(n)

    if slices and usable:
        ref = normals[0]
        consistent = all(float(np.dot(ref, n)) > 0.99 for n in normals[1:])
        if consistent:
            positions = [project_position(s.ipp, ref) for s in slices]
            ordered_positions = sorted(p for p in positions if p is not None)
            gaps = np.diff(np.asarray(ordered_positions)) if len(ordered_positions) >= 2 else np.array([])
            gap_cv = (
                float(np.std(gaps) / np.mean(gaps))
                if gaps.size and float(np.mean(gaps)) > 0
                else None
            )
            duplicates = int(np.sum(gaps < dup_epsilon_mm)) if gaps.size else 0
            return SeriesGeometry(
                ordering_method=ORDERING_SPATIAL,
                normal=(float(ref[0]), float(ref[1]), float(ref[2])),
                positions_mm=list(positions),
                median_gap_mm=float(np.median(gaps)) if gaps.size else None,
                gap_cv=gap_cv,
                duplicate_positions=duplicates,
                span_mm=(
                    float(ordered_positions[-1] - ordered_positions[0])
                    if len(ordered_positions) >= 2
                    else None
                ),
                laterality=_majority_non_empty([s.extra.get("laterality") for s in slices]),
                patient_position=_majority_non_empty([s.extra.get("patient_position") for s in slices]),
            )

    if all(s.instance_number is not None for s in slices):
        method = ORDERING_INSTANCE
    else:
        method = ORDERING_SOP
    return SeriesGeometry(
        ordering_method=method,
        normal=None,
        positions_mm=[None] * len(slices),
        median_gap_mm=None,
        gap_cv=None,
        duplicate_positions=0,
        span_mm=None,
        laterality=_majority_non_empty([s.extra.get("laterality") for s in slices]),
        patient_position=_majority_non_empty([s.extra.get("patient_position") for s in slices]),
    )


def order_slices(
    slices: list[SliceRow],
) -> tuple[list[SliceRow], SeriesGeometry]:
    """Return slices in canonical anatomical order plus the geometry audit.

    Canonical order (Stage-1 recommendation R3):
      spatial   : ascending ImagePositionPatient . (IOP_row x IOP_col)
      fallback  : InstanceNumber ascending, then SOPInstanceUID ascending.
    geometry.positions_mm is realigned to match the RETURNED order so that
    positions[i] always describes ordered_slices[i].
    """
    geometry = build_geometry(slices)
    if geometry.ordering_method == ORDERING_SPATIAL:
        assert geometry.normal is not None
        ref = np.asarray(geometry.normal, dtype=np.float64)

        def sort_key(t):
            pos = project_position(t.ipp, ref)
            return (pos if pos is not None else 0.0, t.sop_uid)

        ordered = sorted(slices, key=sort_key)
        geometry.positions_mm = [
            project_position(s.ipp, ref) for s in ordered
        ]
        return ordered, geometry

    if geometry.ordering_method == ORDERING_INSTANCE:
        ordered = sorted(slices, key=lambda s: (s.instance_number or 0, s.sop_uid))
    else:
        ordered = sorted(slices, key=lambda s: s.sop_uid)
    return ordered, geometry
