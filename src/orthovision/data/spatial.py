"""Slice ordering and anatomical plane from DICOM spatial tags.

Method (documented for audit):

Slice direction
    ImageOrientationPatient is six direction cosines: row (X_image) then
    column (Y_image) in the DICOM patient LPS frame.
    The slice normal n = normalize(row × column).
    Slice coordinate = ImagePositionPatient · n.

Ordering
    If every instance has IPP and a usable IOP, sort by that coordinate.
    Ties broken by InstanceNumber then SOPInstanceUID (deterministic).
    If spatial tags are missing on any slice, fall back to InstanceNumber,
    then SOPInstanceUID. The fallback is recorded on OrderingAudit.

Plane
    |n| aligned with patient X → sagittal, Y → coronal, Z → axial.
    If the dominant component is below `plane_min_alignment`, or two axes
    are too close (`plane_min_axis_separation`), plane is "unknown".
    SeriesDescription is parsed only as a hint; it never overrides spatial
    geometry, and is never used as the official plane.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from orthovision.data.models import OrderingAudit, Plane, PlaneAudit, SliceRecord

_DESC_HINTS: tuple[tuple[str, Plane], ...] = (
    ("SAG", "sagittal"),
    ("COR", "coronal"),
    ("AX", "axial"),
    ("TRA", "axial"),
)


def slice_normal(iop: Sequence[float] | None) -> np.ndarray | None:
    if iop is None or len(iop) < 6:
        return None
    row = np.asarray(iop[:3], dtype=np.float64)
    col = np.asarray(iop[3:6], dtype=np.float64)
    n = np.cross(row, col)
    norm = float(np.linalg.norm(n))
    if norm < 1e-8:
        return None
    return n / norm


def project_position(ipp: Sequence[float] | None, normal: np.ndarray | None) -> float | None:
    if ipp is None or normal is None or len(ipp) != 3:
        return None
    return float(np.dot(np.asarray(ipp, dtype=np.float64), normal))


def spatial_ready(slices: Sequence[SliceRecord]) -> tuple[bool, np.ndarray | None, list[str]]:
    notes: list[str] = []
    if not slices:
        return False, None, ["empty series"]
    normals: list[np.ndarray] = []
    missing = 0
    for sl in slices:
        n = slice_normal(sl.spatial.image_orientation_patient)
        ipp = sl.spatial.image_position_patient
        if n is None or ipp is None:
            missing += 1
            continue
        normals.append(n)
    if missing:
        notes.append(f"{missing}/{len(slices)} instances missing IPP and/or IOP")
    if len(normals) != len(slices):
        notes.append("spatial ordering unavailable; not every instance has IPP+IOP")
        return False, None, notes
    ref = normals[0]
    for n in normals[1:]:
        if abs(float(np.dot(ref, n))) < 0.99:
            notes.append("inconsistent slice normals; spatial sort not applied")
            return False, None, notes
    # Keep a consistent sign: first normal
    return True, ref, notes


def order_slices(slices: list[SliceRecord]) -> tuple[list[SliceRecord], OrderingAudit]:
    ok, normal, notes = spatial_ready(slices)
    positions: list[float | None] = []
    n_tuple = tuple(float(x) for x in normal) if normal is not None else None

    if ok and normal is not None:
        decorated: list[tuple[float, int, str, SliceRecord]] = []
        for sl in slices:
            pos = project_position(sl.spatial.image_position_patient, normal)
            positions.append(pos)
            sl.slice_position = pos
            inst = sl.instance_number if sl.instance_number is not None else 0
            decorated.append((pos if pos is not None else 0.0, inst, sl.sop_instance_uid, sl))
        decorated.sort(key=lambda t: (t[0], t[1], t[2]))
        ordered = [t[3] for t in decorated]
        notes.append("ordered by ImagePositionPatient along slice normal")
        audit = OrderingAudit(
            method="spatial",
            spatial_available=True,
            slice_normal=n_tuple,
            positions=[t[0] for t in decorated],
            notes=notes,
        )
        return ordered, audit

    for sl in slices:
        sl.slice_position = None
        positions.append(None)

    if all(sl.instance_number is not None for sl in slices):
        ordered = sorted(slices, key=lambda s: (s.instance_number or 0, s.sop_instance_uid))
        notes.append("FALLBACK: ordered by InstanceNumber (spatial metadata incomplete)")
        method: str = "instance_number"
    else:
        ordered = sorted(slices, key=lambda s: s.sop_instance_uid)
        notes.append("FALLBACK: ordered by SOPInstanceUID (no spatial tags, InstanceNumber incomplete)")
        method = "sop_instance_uid"

    audit = OrderingAudit(
        method=method,  # type: ignore[arg-type]
        spatial_available=False,
        slice_normal=n_tuple,
        positions=positions,
        notes=notes,
    )
    return ordered, audit


def description_plane_hint(series_description: str | None) -> Plane | None:
    if not series_description:
        return None
    text = series_description.upper()
    hits: list[Plane] = []
    for token, plane in _DESC_HINTS:
        if re.search(rf"\b{token}", text) or token in text.replace(" ", ""):
            if plane not in hits:
                hits.append(plane)
    if len(hits) == 1:
        return hits[0]
    return None


def detect_plane(
    slices: Sequence[SliceRecord],
    *,
    series_description: str | None = None,
    min_alignment: float = 0.8,
    min_axis_separation: float = 0.2,
) -> PlaneAudit:
    hint = description_plane_hint(series_description)
    notes: list[str] = []
    ok, normal, spatial_notes = spatial_ready(list(slices))
    notes.extend(spatial_notes)
    if hint is not None:
        notes.append(f"SeriesDescription hint={hint} (not used as official plane)")

    if not ok or normal is None:
        notes.append("plane unknown: spatial orientation unavailable or inconsistent")
        return PlaneAudit(
            plane="unknown",
            method="unknown",
            slice_normal=None,
            alignment=None,
            description_hint=hint,
            notes=notes,
        )

    abs_n = np.abs(normal)
    axes = ("sagittal", "coronal", "axial")
    alignment = {axes[i]: float(abs_n[i]) for i in range(3)}
    order = np.argsort(abs_n)
    best_i = int(order[-1])
    best = float(abs_n[best_i])
    second = float(abs_n[int(order[-2])])
    n_tuple = (float(normal[0]), float(normal[1]), float(normal[2]))

    if best < min_alignment or (best - second) < min_axis_separation:
        notes.append(
            f"plane unknown: oblique/ambiguous alignment={alignment} "
            f"(need max>={min_alignment} and separation>={min_axis_separation})"
        )
        return PlaneAudit(
            plane="unknown",
            method="spatial",
            slice_normal=n_tuple,
            alignment=alignment,
            description_hint=hint,
            notes=notes,
        )

    plane: Plane = axes[best_i]  # type: ignore[assignment]
    notes.append(f"plane={plane} from slice normal in LPS (X=sagittal, Y=coronal, Z=axial)")
    if hint is not None and hint != plane:
        notes.append(f"WARNING: SeriesDescription hint {hint} disagrees with spatial plane {plane}")
    return PlaneAudit(
        plane=plane,
        method="spatial",
        slice_normal=n_tuple,
        alignment=alignment,
        description_hint=hint,
        notes=notes,
    )
