"""Stage-2 pipeline: study -> selected series -> 2.5D samples + provenance.

Orchestration only; each step lives in its own module and is independently
testable. Raw DICOM files are opened read-only, never modified.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from orthovision.data.pixels import PixelDecodeError, decode_path
from orthovision.preprocessing.cache import SampleCache, series_cache_key
from orthovision.preprocessing.config import PreprocessConfig
from orthovision.preprocessing.discovery import SeriesInfo
from orthovision.preprocessing.intensity import prepare_triplet
from orthovision.preprocessing.qc import (
    QCIssue,
    SeriesQC,
    SeriesQCError,
    check_pixel_array,
    enforce,
    validate_series,
)
from orthovision.preprocessing.resize import stack_triplet_channels
from orthovision.preprocessing.sampling import SamplingError, plan_series_samples
from orthovision.preprocessing.selection import SelectionReport, select_study_series

log = logging.getLogger("orthovision.preprocessing.pipeline")


@dataclass
class SampleRecord:
    image: np.ndarray  # [3, H, W] float32
    metadata: dict


@dataclass
class StudyResult:
    study_uid: str
    selections: dict[str, SelectionReport]
    samples: list[SampleRecord] = field(default_factory=list)
    skipped_series: list[dict] = field(default_factory=list)


class StudyPreprocessor:
    def __init__(
        self,
        cfg: PreprocessConfig,
        *,
        data_root: Path | str,
        cache_dir: Path | str | None = None,
    ):
        self.cfg = cfg
        self.data_root = Path(data_root)
        cache_dir = cache_dir if cache_dir is not None else cfg.cache_dir
        self.cache = SampleCache(cache_dir, enabled=cfg.cache_enabled)

    def process_study(
        self,
        study_uid: str,
        series_infos: list[SeriesInfo],
    ) -> StudyResult:
        selections = select_study_series(series_infos, self.cfg)
        result = StudyResult(study_uid=study_uid, selections=selections)

        for plane in self.cfg.planes:
            report = selections.get(plane)
            if report is None or report.selected_series_uid is None:
                log.info("study %s plane %s: no usable candidate", study_uid[-12:], plane)
                continue
            info = next(s for s in series_infos if s.series_uid == report.selected_series_uid)
            qc = validate_series(info, self.cfg)
            if not qc.usable:
                self._handle_unusable(result, plane, qc)
                continue

            try:
                samples, skip_notes = self.process_series(info)
            except SeriesQCError:
                raise
            except (PixelDecodeError, SamplingError) as exc:
                if self.cfg.mode == "strict":
                    raise
                log.warning("study %s plane %s: %s", study_uid[-12:], plane, exc)
                result.skipped_series.append({"series_uid": info.series_uid, "plane": plane, "reason": str(exc)})
                continue

            result.samples.extend(samples)
            for note in skip_notes:
                result.skipped_series.append({"series_uid": info.series_uid, "plane": plane, **note})
        return result

    def _handle_unusable(self, result: StudyResult, plane: str, qc: SeriesQC) -> None:
        detail = "; ".join(f"{i.code}: {i.detail}" for i in qc.issues)
        if self.cfg.mode == "strict":
            raise SeriesQCError(qc)
        log.warning("study %s plane %s unusable series %s: %s",
                    result.study_uid[-12:], plane, qc.series_uid[-12:], detail)
        result.skipped_series.append(
            {"series_uid": qc.series_uid, "plane": plane, "reason": detail}
        )

    def process_series(self, info: SeriesInfo) -> tuple[list[SampleRecord], list[dict]]:
        key = series_cache_key(
            self.cfg,
            info.series_uid,
            [s.sop_uid for s in info.slices_ordered],
        )
        cached = self.cache.get(key)
        if cached is not None:
            payload = json.loads(cached.metadata_json)
            records = [
                SampleRecord(image=img, metadata=meta)
                for img, meta in zip(cached.images, payload["samples"])
            ]
            return records, []

        positions = (
            [float(p) if p is not None else None for p in info.geometry.positions_mm]
            if info.geometry.ordering_method == "spatial"
            else None
        )
        plans = plan_series_samples(info.n_slices, positions, self.cfg)
        notes: list[dict] = []
        requested = len(plans)

        images: list[np.ndarray] = []
        metas: list[dict] = []
        for plan in plans:
            raw_channels: list[np.ndarray] = []
            channel_prov: list[dict] = []
            photometrics: list[str | None] = []
            failed = False
            for idx in plan.slice_indices:
                row = info.slices_ordered[idx]
                try:
                    payload, prov = self._load_raw(row)
                except PixelDecodeError as exc:
                    issue = QCIssue("BLOCKING", "unreadable_dicom", f"{row.sop_uid}: {exc}")
                    if self.cfg.mode == "strict":
                        raise
                    log.warning("series %s: skipping sample (%s)", info.series_uid[-12:], issue.detail)
                    notes.append({"reason": f"unreadable slice {idx}", "detail": issue.detail})
                    failed = True
                    break
                pixel_issue = check_pixel_array(payload.rescaled, row.sop_uid)
                if pixel_issue is not None and pixel_issue.severity == "BLOCKING":
                    if self.cfg.mode == "strict":
                        raise ValueError(pixel_issue.detail)
                    log.warning("series %s: %s", info.series_uid[-12:], pixel_issue.detail)
                    notes.append({"reason": "invalid_pixels", "detail": pixel_issue.detail})
                    failed = True
                    break
                raw_channels.append(payload.rescaled)
                photometrics.append(row.photometric or payload.photometric)
                channel_prov.append(prov)
            if failed or len(raw_channels) != 3:
                continue

            channels, norm_stats = prepare_triplet(
                raw_channels, photometrics=photometrics, cfg=self.cfg
            )

            stacked, resize_metas = stack_triplet_channels(channels, self.cfg)
            meta = {
                "study_uid": info.study_uid,
                "series_uid": info.series_uid,
                "plane_csv": info.plane_csv,
                "plane_derived": info.plane_derived,
                "laterality": info.laterality,
                "patient_position": info.patient_position,
                "ordering_method": info.ordering_method,
                "slice_normal_lps": list(info.orientation_normal) if info.orientation_normal else None,
                "center_slice_index": plan.center_index,
                "slice_indices": list(plan.slice_indices),
                "physical_positions_mm": list(plan.physical_positions_mm),
                "channel_roles": ["previous", "center", "next"],
                "sop_instance_uids": [info.slices_ordered[i].sop_uid for i in plan.slice_indices],
                "source_paths": [info.slices_ordered[i].path for i in plan.slice_indices],
                "rescale_slopes": [p["rescale_slope"] for p in channel_prov],
                "normalization_stats": [
                    {
                        "method": s.method,
                        "scope": s.scope,
                        "clip_low": s.clip_low,
                        "clip_high": s.clip_high,
                        "mean": s.mean,
                        "std": s.std,
                        "inverted_monochrome1": s.inverted_monochrome1,
                    }
                    for s in norm_stats
                ],
                "resize_meta": [m.__dict__ for m in resize_metas],
                "config_fingerprint": self.cfg.fingerprint(),
            }
            images.append(stacked)
            metas.append(meta)

        if images:
            arr = np.stack(images, axis=0).astype(np.float32)
            self.cache.put(key, arr, json.dumps({"samples": metas}))
        else:
            arr = np.empty((0,), dtype=np.float32)

        if requested < self.cfg.num_samples_per_plane:
            notes.append(
                {
                    "reason": "insufficient_samples_for_request",
                    "detail": f"{requested} locations produced {len(images)} triplets "
                    f"(requested {self.cfg.num_samples_per_plane})",
                }
            )
        return [SampleRecord(image=img, metadata=meta) for img, meta in zip(images, metas)], notes

    def _load_raw(self, row):
        path = self.data_root / row.path
        payload = decode_path(path)
        prov = {
            "rescale_slope": payload.rescale_slope,
            "rescale_intercept": payload.rescale_intercept,
        }
        return payload, prov
