"""Machine-readable manifests (CSV + JSON). No pixel arrays."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from orthovision.data.catalog import KneeMRIDataset, summarize_study
from orthovision.data.logutil import get_logger
from orthovision.data.validate import series_slice_spacing

log = get_logger("manifest")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def series_rows(ds: KneeMRIDataset) -> list[dict]:
    ds.ensure_scanned()
    rows: list[dict] = []
    root = ds.config.dataset_root
    for study, ser in ds.iter_series():
        spat = ser.representative_spatial()
        paths = ser.ordered_slice_paths()
        common = None
        if paths:
            try:
                common = Path(paths[0]).parent
            except Exception:
                common = None
        val = ser.validation
        rows.append(
            {
                "study_id": study.study_id,
                "series_id": ser.series_id,
                "series_description": ser.series_description,
                "series_number": ser.series_number,
                "modality": ser.modality,
                "body_part_examined": ser.body_part_examined,
                "protocol_name": ser.protocol_name,
                "sequence_name": ser.sequence_name,
                "plane": ser.plane,
                "slice_count": ser.slice_count,
                "rows": spat.rows if spat else None,
                "columns": spat.columns if spat else None,
                "pixel_spacing": list(spat.pixel_spacing) if spat and spat.pixel_spacing else None,
                "slice_spacing": series_slice_spacing(ser),
                "slice_thickness": spat.slice_thickness if spat else None,
                "orientation": list(spat.image_orientation_patient) if spat and spat.image_orientation_patient else None,
                "transfer_syntax_uid": ser.transfer_syntax_uid,
                "ordering_method": ser.ordering_audit.method if ser.ordering_audit else None,
                "ordering_notes": " | ".join(ser.ordering_audit.notes) if ser.ordering_audit else None,
                "validation_status": val.status if val else None,
                "validation_messages": " | ".join(val.messages) if val else None,
                "source_dir": _rel(common, root) if common else None,
                "example_path": _rel(paths[0], root) if paths else None,
            }
        )
    return rows


def write_manifests(ds: KneeMRIDataset, dest: Path | None = None) -> dict[str, Path]:
    ds.ensure_scanned()
    dest = dest or ds.config.manifest_dir
    dest.mkdir(parents=True, exist_ok=True)

    rows = series_rows(ds)
    csv_path = dest / "series_manifest.csv"
    json_path = dest / "series_manifest.json"
    issues_path = dest / "file_issues.csv"
    summary_path = dest / "scan_summary.json"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    issue_rows = [
        {
            "path": i.path,
            "stage": i.stage,
            "message": i.message,
            "exception_type": i.exception_type,
            "kind": i.kind,
        }
        for i in ds.issues
    ]
    pd.DataFrame(issue_rows).to_csv(issues_path, index=False)

    study_summaries = [summarize_study(s).__dict__ for s in ds.studies.values()]
    # SeriesSummary dataclasses inside
    def _ser(s):
        out = dict(s.__dict__)
        out["series"] = [x.__dict__ for x in s.series]
        return out

    payload = {
        "stats": ds.stats.__dict__,
        "dataset_root": str(ds.config.dataset_root),
        "studies": [_ser(summarize_study(s)) for s in ds.studies.values()],
        "note": "Pixel arrays are not stored in the manifest.",
    }
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("Wrote manifests to %s (%s series rows)", dest, len(rows))
    return {"csv": csv_path, "json": json_path, "issues": issues_path, "summary": summary_path}
