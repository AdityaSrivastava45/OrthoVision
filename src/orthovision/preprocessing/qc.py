"""Quality-control checks for series and samples.

Permissive mode collects issues and skips unusable series/samples with clear
logging. Strict mode raises SeriesQCError on any BLOCKING issue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from orthovision.preprocessing.config import PreprocessConfig
from orthovision.preprocessing.discovery import SeriesInfo

log = logging.getLogger("orthovision.preprocessing.qc")


@dataclass(frozen=True)
class QCIssue:
    severity: str  # "BLOCKING" | "WARNING" | "INFO"
    code: str
    detail: str


@dataclass
class SeriesQC:
    series_uid: str
    usable: bool
    issues: list[QCIssue] = field(default_factory=list)

    def warnings(self) -> list[QCIssue]:
        return [i for i in self.issues if i.severity != "BLOCKING"]


class SeriesQCError(RuntimeError):
    def __init__(self, qc: "SeriesQC"):
        lines = "\n".join(f"  [{i.severity}] {i.code}: {i.detail}" for i in qc.issues)
        super().__init__(f"series {qc.series_uid} failed strict QC:\n{lines}")
        self.qc = qc


def _check(info: SeriesInfo, cfg: PreprocessConfig) -> list[QCIssue]:
    issues: list[QCIssue] = []
    geometry = info.geometry

    if info.n_slices == 0:
        issues.append(QCIssue("BLOCKING", "empty_series", "series has no DICOM instances"))
        return issues

    if info.modality not in (None, "MR"):
        issues.append(QCIssue("WARNING", "unexpected_modality", f"Modality={info.modality}"))

    if info.n_slices < cfg.min_slices_usable:
        issues.append(
            QCIssue(
                "BLOCKING",
                "insufficient_slices",
                f"{info.n_slices} slices < min {cfg.min_slices_usable}",
            )
        )

    if geometry.ordering_method != "spatial":
        issues.append(
            QCIssue(
                "WARNING",
                "missing_geometry",
                f"ordering fell back to {geometry.ordering_method}; no IPP+IOP ordering",
            )
        )
    else:
        if geometry.duplicate_positions > 0:
            issues.append(
                QCIssue(
                    "WARNING",
                    "duplicate_positions",
                    f"{geometry.duplicate_positions} near-duplicate positions "
                    f"(eps={cfg.duplicate_position_epsilon_mm} mm)",
                )
            )
        if geometry.gap_cv is not None and geometry.gap_cv > cfg.unusual_spacing_cv:
            issues.append(
                QCIssue(
                    "WARNING",
                    "inconsistent_spacing",
                    f"gap CV={geometry.gap_cv:.3f} > {cfg.unusual_spacing_cv}",
                )
            )

    sizes = {(s.rows, s.columns) for s in info.slices_ordered}
    if any(None in s for s in sizes):
        issues.append(QCIssue("WARNING", "inconsistent_dimensions", "some slices missing Rows/Columns"))
    complete_sizes = {s for s in sizes if None not in s}
    if len(complete_sizes) > 1:
        issues.append(
            QCIssue("BLOCKING", "inconsistent_dimensions", f"mixed matrix sizes within series: {sorted(complete_sizes)}")
        )

    agreement = info.plane_agrees()
    if agreement is False:
        issues.append(
            QCIssue(
                "WARNING",
                "plane_mismatch",
                f"CSV plane {info.plane_csv} disagrees with DICOM-derived plane {info.plane_derived}",
            )
        )
    elif agreement is None and info.plane_derived is None:
        issues.append(
            QCIssue("WARNING", "unusual_orientation", "anatomical plane could not be derived from IOP")
        )

    return issues


def validate_series(info: SeriesInfo, cfg: PreprocessConfig) -> SeriesQC:
    issues = _check(info, cfg)
    blocking = [i for i in issues if i.severity == "BLOCKING"]
    qc = SeriesQC(series_uid=info.series_uid, usable=not blocking, issues=issues)
    for issue in qc.warnings():
        log.warning("[series %s] %s: %s", info.series_uid[-12:], issue.code, issue.detail)
    return qc


def enforce(qc: SeriesQC, cfg: PreprocessConfig) -> None:
    if cfg.mode == "strict":
        raise SeriesQCError(qc)


def check_pixel_array(arr: np.ndarray, sop_uid: str) -> QCIssue | None:
    if arr.size == 0:
        return QCIssue("BLOCKING", "invalid_pixels", f"{sop_uid}: empty pixel array")
    if not np.isfinite(arr).all():
        return QCIssue("BLOCKING", "invalid_pixels", f"{sop_uid}: non-finite pixel values")
    if float(np.max(arr)) == float(np.min(arr)):
        return QCIssue("WARNING", "constant_pixels", f"{sop_uid}: constant-valued image")
    return None
