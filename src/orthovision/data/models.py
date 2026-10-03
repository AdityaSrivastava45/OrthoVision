"""In-memory data model: Dataset → Study → Series → Slice.

Filesystem paths stay inside this layer. Downstream ML code should use
study/series UIDs plus decoded arrays and SpatialMetadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np

Plane = Literal["sagittal", "coronal", "axial", "unknown"]
OrderingMethod = Literal["spatial", "instance_number", "sop_instance_uid"]
ValidationStatus = Literal["VALID", "WARNING", "INVALID"]
FileKind = Literal["dicom", "not_dicom", "unreadable"]


class ScanStage(str, Enum):
    DISCOVERY = "discovery"
    DECODE = "decode"
    GROUPING = "grouping"
    ORDERING = "ordering"
    VALIDATION = "validation"


@dataclass
class FileIssue:
    path: str
    stage: str
    message: str
    exception_type: str | None = None
    kind: FileKind | None = None


@dataclass(frozen=True)
class SpatialMetadata:
    rows: int | None
    columns: int | None
    pixel_spacing: tuple[float, float] | None
    slice_thickness: float | None
    spacing_between_slices: float | None
    image_orientation_patient: tuple[float, ...] | None
    image_position_patient: tuple[float, float, float] | None


@dataclass
class OrderingAudit:
    method: OrderingMethod
    spatial_available: bool
    slice_normal: tuple[float, float, float] | None
    positions: list[float | None]
    notes: list[str] = field(default_factory=list)


@dataclass
class PlaneAudit:
    plane: Plane
    method: Literal["spatial", "unknown"]
    slice_normal: tuple[float, float, float] | None
    alignment: dict[str, float] | None
    description_hint: Plane | None
    notes: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    status: ValidationStatus
    messages: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.messages.append(message)


@dataclass
class SliceRecord:
    """One DICOM instance. Pixel arrays are decoded on demand, not stored here."""

    sop_instance_uid: str
    source_path: Path
    instance_number: int | None
    transfer_syntax_uid: str | None
    spatial: SpatialMetadata
    slice_position: float | None
    number_of_frames: int | None
    metadata: dict[str, Any] = field(default_factory=dict)
    read_error: str | None = None

    @property
    def study_id(self) -> str | None:
        return self.metadata.get("StudyInstanceUID")

    @property
    def series_id(self) -> str | None:
        return self.metadata.get("SeriesInstanceUID")


@dataclass
class Series:
    study_instance_uid: str
    series_instance_uid: str
    slices: list[SliceRecord]
    series_description: str | None = None
    series_number: int | None = None
    modality: str | None = None
    body_part_examined: str | None = None
    protocol_name: str | None = None
    sequence_name: str | None = None
    transfer_syntax_uid: str | None = None
    plane_audit: PlaneAudit | None = None
    ordering_audit: OrderingAudit | None = None
    validation: ValidationResult | None = None

    @property
    def series_id(self) -> str:
        return self.series_instance_uid

    @property
    def plane(self) -> Plane:
        if self.plane_audit is None:
            return "unknown"
        return self.plane_audit.plane

    @property
    def slice_count(self) -> int:
        return len(self.slices)

    def representative_spatial(self) -> SpatialMetadata | None:
        if not self.slices:
            return None
        return self.slices[0].spatial

    def ordered_slice_paths(self) -> list[Path]:
        return [s.source_path for s in self.slices]


@dataclass
class SeriesSummary:
    series_id: str
    series_description: str | None
    plane: Plane
    slice_count: int
    rows: int | None
    columns: int | None
    pixel_spacing: tuple[float, float] | None
    slice_spacing: float | None
    orientation: tuple[float, ...] | None
    validation_status: ValidationStatus
    validation_messages: list[str]
    ordering_method: OrderingMethod | None


@dataclass
class Study:
    study_instance_uid: str
    series: list[Series]
    extra_issues: list[FileIssue] = field(default_factory=list)

    @property
    def study_id(self) -> str:
        return self.study_instance_uid

    def get_series(self, series_id: str) -> Series:
        for ser in self.series:
            if ser.series_id == series_id:
                return ser
        raise KeyError(f"Series not found: {series_id}")


@dataclass
class StudySummary:
    study_id: str
    n_series: int
    n_valid_series: int
    n_warning_series: int
    n_invalid_series: int
    available_planes: list[Plane]
    total_instances: int
    validation_warnings: list[str]
    series: list[SeriesSummary]


@dataclass
class ScanStats:
    files_seen: int = 0
    dicom_valid: int = 0
    dicom_unreadable: int = 0
    not_dicom: int = 0
    n_studies: int = 0
    n_series: int = 0
    n_valid_series: int = 0
    n_warning_series: int = 0
    n_invalid_series: int = 0
    n_file_issues: int = 0


def empty_spatial() -> SpatialMetadata:
    return SpatialMetadata(
        rows=None,
        columns=None,
        pixel_spacing=None,
        slice_thickness=None,
        spacing_between_slices=None,
        image_orientation_patient=None,
        image_position_patient=None,
    )


# Used only for type checkers that import numpy from this module.
Array = np.ndarray
