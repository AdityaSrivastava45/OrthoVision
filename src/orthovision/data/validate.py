"""Series-level quality checks. Status is VALID / WARNING / INVALID with reasons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from orthovision.data.config import DataConfig
from orthovision.data.models import Series, ValidationResult, ValidationStatus
from orthovision.data.spatial import project_position, slice_normal


def _status_from_flags(has_invalid: bool, has_warning: bool) -> ValidationStatus:
    if has_invalid:
        return "INVALID"
    if has_warning:
        return "WARNING"
    return "VALID"


def validate_series(series: Series, cfg: DataConfig) -> ValidationResult:
    messages: list[str] = []
    invalid = False
    warning = False

    n = series.slice_count
    if n == 0:
        invalid = True
        messages.append("empty series (no DICOM instances)")
        return ValidationResult(status="INVALID", messages=messages)

    unreadable = [s for s in series.slices if s.read_error]
    if unreadable:
        invalid = True
        messages.append(f"{len(unreadable)} unreadable instance(s), e.g. {unreadable[0].read_error}")

    rows = {s.spatial.rows for s in series.slices}
    cols = {s.spatial.columns for s in series.slices}
    if None in rows or None in cols:
        warning = True
        messages.append("missing Rows/Columns on at least one instance")
    if len(rows - {None}) > 1 or len(cols - {None}) > 1:
        invalid = True
        messages.append(f"inconsistent dimensions rows={rows} columns={cols}")

    spacings = {s.spatial.pixel_spacing for s in series.slices}
    if None in spacings:
        warning = True
        messages.append("missing PixelSpacing on at least one instance")
    elif len(spacings) > 1:
        warning = True
        messages.append(f"inconsistent PixelSpacing values: {spacings}")

    iops = [s.spatial.image_orientation_patient for s in series.slices]
    if any(iop is None for iop in iops):
        warning = True
        messages.append("missing ImageOrientationPatient on at least one instance")
    else:
        ref = np.asarray(iops[0], dtype=np.float64)
        for iop in iops[1:]:
            other = np.asarray(iop, dtype=np.float64)
            if float(np.max(np.abs(ref - other))) > cfg.orientation_epsilon:
                invalid = True
                messages.append("inconsistent ImageOrientationPatient across slices")
                break

    ipps = [s.spatial.image_position_patient for s in series.slices]
    if any(ipp is None for ipp in ipps):
        warning = True
        messages.append("missing ImagePositionPatient on at least one instance")

    if series.ordering_audit and series.ordering_audit.method != "spatial":
        warning = True
        messages.append(f"slice order used fallback method={series.ordering_audit.method}")

    positions = _positions(series)
    if positions is not None and len(positions) >= 2:
        diffs = np.diff(positions)
        abs_d = np.abs(diffs)
        if np.any(abs_d < cfg.duplicate_position_epsilon_mm):
            warning = True
            messages.append("duplicate or near-duplicate slice positions")
        if float(np.mean(abs_d)) > 0:
            cv = float(np.std(abs_d) / np.mean(abs_d))
            if cv > cfg.unusual_spacing_cv:
                warning = True
                messages.append(f"unusual slice spacing variation CV={cv:.3f} > {cfg.unusual_spacing_cv}")
        computed = float(np.median(abs_d)) if abs_d.size else None
        tagged = series.slices[0].spatial.spacing_between_slices
        if computed is not None and tagged is not None and tagged > 0:
            if abs(computed - tagged) / tagged > 0.25:
                warning = True
                messages.append(
                    f"SpacingBetweenSlices tag={tagged:g} disagrees with median position delta={computed:g}"
                )
    elif positions is None:
        warning = True
        messages.append("could not compute slice positions for spacing checks")

    if n < cfg.min_slices_warning:
        warning = True
        messages.append(f"few slices ({n} < {cfg.min_slices_warning})")

    if series.plane == "unknown":
        warning = True
        messages.append("anatomical plane unknown")

    frames = {s.number_of_frames for s in series.slices if s.number_of_frames and s.number_of_frames > 1}
    if frames:
        warning = True
        messages.append(f"multi-frame DICOM detected NumberOfFrames={frames} (not expanded in this milestone)")

    ts = {s.transfer_syntax_uid for s in series.slices}
    if len(ts) > 1:
        warning = True
        messages.append(f"mixed transfer syntaxes in series: {ts}")

    if series.plane_audit and series.plane_audit.description_hint and series.plane_audit.plane not in ("unknown", series.plane_audit.description_hint):
        warning = True
        messages.append("SeriesDescription plane hint disagrees with spatial plane")

    if not messages:
        messages.append("all inspected checks passed")

    return ValidationResult(status=_status_from_flags(invalid, warning), messages=messages)


def _positions(series: Series) -> np.ndarray | None:
    if series.ordering_audit and series.ordering_audit.method == "spatial":
        vals = [p for p in series.ordering_audit.positions if p is not None]
        if len(vals) == len(series.slices):
            return np.asarray(vals, dtype=np.float64)
    if not series.slices:
        return None
    n = slice_normal(series.slices[0].spatial.image_orientation_patient)
    if n is None:
        return None
    out: list[float] = []
    for sl in series.slices:
        p = project_position(sl.spatial.image_position_patient, n)
        if p is None:
            return None
        out.append(p)
    return np.asarray(out, dtype=np.float64)


def series_slice_spacing(series: Series) -> float | None:
    tagged = series.slices[0].spatial.spacing_between_slices if series.slices else None
    pos = _positions(series)
    if pos is not None and len(pos) >= 2:
        return float(np.median(np.abs(np.diff(pos))))
    return tagged
