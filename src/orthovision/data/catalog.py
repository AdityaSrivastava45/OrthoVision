"""High-level catalog: filesystem → studies/series without leaking paths to ML code."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from orthovision.data.config import DataConfig, load_data_config
from orthovision.data.discovery import HeaderRecord, discover_dicom_headers
from orthovision.data.logutil import configure_logging, get_logger
from orthovision.data.models import (
    FileIssue,
    ScanStats,
    Series,
    SeriesSummary,
    SliceRecord,
    Study,
    StudySummary,
)
from orthovision.data.spatial import detect_plane, order_slices
from orthovision.data.validate import series_slice_spacing, validate_series

log = get_logger("catalog")

MISSING_STUDY = "MISSING_STUDY"
MISSING_SERIES = "MISSING_SERIES"


def _slice_from_header(h: HeaderRecord) -> SliceRecord:
    return SliceRecord(
        sop_instance_uid=h.sop_uid or str(h.path),
        source_path=h.path,
        instance_number=h.instance_number,
        transfer_syntax_uid=h.transfer_syntax_uid,
        spatial=h.spatial,
        slice_position=None,
        number_of_frames=h.number_of_frames,
        metadata=dict(h.metadata),
    )


def _series_from_headers(study_uid: str, series_uid: str, headers: list[HeaderRecord], cfg: DataConfig) -> Series:
    slices = [_slice_from_header(h) for h in headers]
    first = headers[0]
    desc = first.metadata.get("SeriesDescription")
    if isinstance(desc, (int, float)):
        desc = str(desc)
    series = Series(
        study_instance_uid=study_uid,
        series_instance_uid=series_uid,
        slices=slices,
        series_description=desc if isinstance(desc, str) else None,
        series_number=first.metadata.get("SeriesNumber") if isinstance(first.metadata.get("SeriesNumber"), int) else None,
        modality=first.metadata.get("Modality") if isinstance(first.metadata.get("Modality"), str) else None,
        body_part_examined=first.metadata.get("BodyPartExamined") if isinstance(first.metadata.get("BodyPartExamined"), str) else None,
        protocol_name=first.metadata.get("ProtocolName") if isinstance(first.metadata.get("ProtocolName"), str) else None,
        sequence_name=first.metadata.get("SequenceName") if isinstance(first.metadata.get("SequenceName"), str) else None,
        transfer_syntax_uid=first.transfer_syntax_uid,
    )
    ordered, order_audit = order_slices(series.slices)
    series.slices = ordered
    series.ordering_audit = order_audit
    series.plane_audit = detect_plane(
        series.slices,
        series_description=series.series_description,
        min_alignment=cfg.plane_min_alignment,
        min_axis_separation=cfg.plane_min_axis_separation,
    )
    series.validation = validate_series(series, cfg)
    return series


def build_studies(
    headers: list[HeaderRecord],
    issues: list[FileIssue],
    cfg: DataConfig,
) -> list[Study]:
    buckets: dict[str, dict[str, list[HeaderRecord]]] = defaultdict(lambda: defaultdict(list))
    for h in headers:
        study = h.study_uid or MISSING_STUDY
        series = h.series_uid or MISSING_SERIES
        buckets[study][series].append(h)

    studies: list[Study] = []
    for study_uid in sorted(buckets):
        series_map = buckets[study_uid]
        series_list = [
            _series_from_headers(study_uid, series_uid, series_map[series_uid], cfg) for series_uid in sorted(series_map)
        ]
        studies.append(Study(study_instance_uid=study_uid, series=series_list, extra_issues=[]))
    return studies


def summarize_study(study: Study) -> StudySummary:
    n_valid = n_warn = n_invalid = 0
    warnings: list[str] = []
    planes: list = []
    summaries: list[SeriesSummary] = []
    total = 0
    for ser in study.series:
        total += ser.slice_count
        st = ser.validation.status if ser.validation else "INVALID"
        if st == "VALID":
            n_valid += 1
        elif st == "WARNING":
            n_warn += 1
            if ser.validation:
                warnings.extend(f"{ser.series_id}: {m}" for m in ser.validation.messages)
        else:
            n_invalid += 1
            if ser.validation:
                warnings.extend(f"{ser.series_id}: {m}" for m in ser.validation.messages)
        if ser.plane not in planes:
            planes.append(ser.plane)
        spat = ser.representative_spatial()
        summaries.append(
            SeriesSummary(
                series_id=ser.series_id,
                series_description=ser.series_description,
                plane=ser.plane,
                slice_count=ser.slice_count,
                rows=spat.rows if spat else None,
                columns=spat.columns if spat else None,
                pixel_spacing=spat.pixel_spacing if spat else None,
                slice_spacing=series_slice_spacing(ser),
                orientation=spat.image_orientation_patient if spat else None,
                validation_status=st,  # type: ignore[arg-type]
                validation_messages=list(ser.validation.messages) if ser.validation else [],
                ordering_method=ser.ordering_audit.method if ser.ordering_audit else None,
            )
        )
    return StudySummary(
        study_id=study.study_id,
        n_series=len(study.series),
        n_valid_series=n_valid,
        n_warning_series=n_warn,
        n_invalid_series=n_invalid,
        available_planes=planes,
        total_instances=total,
        validation_warnings=warnings,
        series=summaries,
    )


class KneeMRIDataset:
    """Inspectable catalog of studies. Pixel arrays are loaded on demand via `orthovision.data.pixels`."""

    def __init__(self, config: DataConfig | None = None, *, config_path: Path | str | None = None):
        self.config = config or load_data_config(config_path)
        self.studies: dict[str, Study] = {}
        self.issues: list[FileIssue] = []
        self.stats = ScanStats()
        self._scanned = False

    def scan(self) -> ScanStats:
        configure_logging(self.config.logging_level, self.config.log_file)
        root = self.config.dataset_root
        log.info("Starting DICOM catalog scan dataset_root=%s", root)
        headers, issues, counts = discover_dicom_headers(root, num_workers=self.config.num_workers)
        studies = build_studies(headers, issues, self.config)
        self.studies = {s.study_id: s for s in studies}
        self.issues = issues
        self.stats = ScanStats(
            files_seen=counts["files_seen"],
            dicom_valid=counts["dicom_valid"],
            dicom_unreadable=counts["dicom_unreadable"],
            not_dicom=counts["not_dicom"],
            n_studies=len(studies),
            n_series=sum(len(s.series) for s in studies),
            n_valid_series=0,
            n_warning_series=0,
            n_invalid_series=0,
            n_file_issues=len(issues),
        )
        for study in studies:
            for ser in study.series:
                st = ser.validation.status if ser.validation else "INVALID"
                if st == "VALID":
                    self.stats.n_valid_series += 1
                elif st == "WARNING":
                    self.stats.n_warning_series += 1
                else:
                    self.stats.n_invalid_series += 1
        self._scanned = True
        log.info(
            "Catalog studies=%s series=%s valid_series=%s warning_series=%s invalid_series=%s file_issues=%s",
            self.stats.n_studies,
            self.stats.n_series,
            self.stats.n_valid_series,
            self.stats.n_warning_series,
            self.stats.n_invalid_series,
            self.stats.n_file_issues,
        )
        return self.stats

    def ensure_scanned(self) -> None:
        if not self._scanned:
            self.scan()

    def study_ids(self) -> list[str]:
        self.ensure_scanned()
        return sorted(self.studies)

    def get_study(self, study_id: str) -> Study:
        self.ensure_scanned()
        if study_id not in self.studies:
            raise KeyError(f"Unknown study_id: {study_id}")
        return self.studies[study_id]

    def study_summary(self, study_id: str) -> StudySummary:
        return summarize_study(self.get_study(study_id))

    def iter_series(self):
        self.ensure_scanned()
        for study in self.studies.values():
            for ser in study.series:
                yield study, ser
