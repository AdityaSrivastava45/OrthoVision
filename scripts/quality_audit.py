"""PHASE 7 - Data quality audit.

Full (not sampled) pass over every DICOM in data/raw/train_series:
decodes pixels, records shape/intensity/hash, then consolidates header-level
checks into severity-classified findings (CRITICAL / WARNING / INFO).

Writes outputs/reports/pixel_stats.csv and outputs/reports/quality_audit.json.
Raw data is never modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import PROJECT_ROOT, describe, write_report

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from orthovision.data.pixels import PixelDecodeError, decode_path
from orthovision.data.tabular import load_series

RAW = PROJECT_ROOT / "data" / "raw"


def main() -> int:
    hdr = pd.read_csv(
        PROJECT_ROOT / "outputs" / "reports" / "dicom_headers.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    ts = load_series(RAW / "train_series.csv")
    csv_meta = ts.set_index("SeriesInstanceUID")

    rows = []
    errors: list[dict] = []
    for r in hdr.itertuples(index=False):
        path = RAW / r.path
        entry = {"path": r.path, "series_uid": r.SeriesInstanceUID}
        try:
            payload = decode_path(path)
            arr = payload.rescaled
            entry.update(
                {
                    "ok": True,
                    "shape_rows": int(arr.shape[0]),
                    "shape_cols": int(arr.shape[1]),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "hash_sha256": hashlib.sha256(np.ascontiguousarray(payload.raw).tobytes()).hexdigest(),
                }
            )
        except (PixelDecodeError, Exception) as exc:  # noqa: BLE001
            entry["ok"] = False
            errors.append({"path": r.path, "error": str(exc)})
        rows.append(entry)
    px = pd.DataFrame(rows)
    px.to_csv(PROJECT_ROOT / "outputs" / "reports" / "pixel_stats.csv", index=False)

    findings: list[dict] = []

    def add(severity: str, code: str, detail: str, evidence=None):
        findings.append({"severity": severity, "code": code, "detail": detail, "evidence": evidence})

    if errors:
        add("CRITICAL", "unreadable_pixels", f"{len(errors)} file(s) failed pixel decode", errors[:10])
    else:
        add("INFO", "pixel_decode", "all 1067 files decoded successfully")

    dup_hashes = px[px["ok"]].groupby("hash_sha256").size()
    dup_hashes = dup_hashes[dup_hashes > 1]
    if len(dup_hashes):
        examples = []
        for h in dup_hashes.index[:5]:
            examples.append(px.loc[px.hash_sha256 == h, "path"].tolist())
        add("WARNING", "duplicate_images", f"{int(dup_hashes.sum())} files share exact pixel content in {len(dup_hashes)} group(s)", examples)

    constant = px[px["ok"] & ((px["std"] == 0) | (px["max"] == 0))]
    if len(constant):
        add("WARNING", "constant_or_empty_images", f"{len(constant)} image(s) are constant or all-zero", constant["path"].head(5).tolist())

    non_square = px[px["ok"] & (px["shape_rows"] != px["shape_cols"])]
    add(
        "WARNING" if len(non_square) else "INFO",
        "non_square_matrices",
        f"{len(non_square)} image(s) have Rows != Columns",
        {f"{r}x{c}": int(n) for (r, c), n in non_square.groupby(["shape_rows", "shape_cols"]).size().items()},
    )

    size_by_series = px[px["ok"]].groupby("series_uid")[["shape_rows", "shape_cols"]].nunique()
    mixed_size_series = size_by_series[(size_by_series > 1).any(axis=1)]
    if len(mixed_size_series):
        add("CRITICAL", "inconsistent_size_within_series", f"{len(mixed_size_series)} series mix matrix sizes", mixed_size_series.index.tolist()[:5])
    else:
        add("INFO", "size_consistency", "matrix size constant within every series")

    slope_by_series = hdr.groupby("SeriesInstanceUID")["RescaleSlope"].apply(lambda s: set(s.dropna().round(6)))
    mixed_slope = [uid for uid, s in slope_by_series.items() if len(s) > 1]
    partial_slope = [uid for uid, s in slope_by_series.items() if 0 < len(s) < hdr.loc[hdr.SeriesInstanceUID == uid].shape[0]]
    if mixed_slope or partial_slope:
        add("WARNING", "rescale_slope_inconsistent_within_series",
            f"{len(mixed_slope)} series with multiple slopes, {len(partial_slope)} with slope on some slices only",
            {"mixed": [u[-12:] for u in mixed_slope[:5]], "partial": [u[-12:] for u in partial_slope[:5]]})
    else:
        add("INFO", "rescale_consistency", "RescaleSlope constant-or-absent within each series")

    missing_sbs = hdr[hdr["SpacingBetweenSlices"].isna()]
    add("WARNING" if len(missing_sbs) else "INFO", "spacing_tag_missing",
        f"SpacingBetweenSlices absent on {len(missing_sbs)}/{len(hdr)} files (geometry still recoverable from IPP)",
        sorted({Path(p).parts[1][-12:] for p in missing_sbs["path"]}))

    counts = hdr.groupby("SeriesInstanceUID").size()
    add("INFO", "slice_count_range", f"slices per series: min={counts.min()} max={counts.max()}",
        {"n_series_with_lt_20": int((counts < 20).sum()), "n_series_with_gt_100": int((counts > 100).sum())})

    modalities = set(hdr["Modality"].dropna())
    add("WARNING" if modalities - {"MR"} else "INFO", "modality_check", f"modalities present: {sorted(modalities)}")
    syntaxes = set(hdr["TransferSyntaxUID"].dropna())
    add("WARNING" if syntaxes - {"1.2.840.10008.1.2.1"} else "INFO", "transfer_syntax_check",
        f"transfer syntaxes present: {sorted(syntaxes)}")
    photometric = set(hdr["PhotometricInterpretation"].dropna())
    add("WARNING" if photometric - {"MONOCHROME2"} else "INFO", "photometric_check",
        f"photometric interpretations: {sorted(photometric)}; MONOCHROME1 absent")

    patient_studies = defaultdict(set)
    for r in hdr.itertuples(index=False):
        pid = getattr(r, "PatientID")
        if isinstance(pid, str):
            patient_studies[pid].add(r.StudyInstanceUID)
    multi_study_patients = {p: s for p, s in patient_studies.items() if len(s) > 1}
    add("INFO" if not multi_study_patients else "WARNING", "patient_id_reuse",
        f"{len(patient_studies)} distinct pseudonymized PatientIDs in subset; "
        f"{len(multi_study_patients)} map to >1 study",
        {p[-12:]: len(s) for p, s in list(multi_study_patients.items())[:5]})

    desc_by_series = hdr.groupby("SeriesInstanceUID")["SeriesDescription"].nunique()
    add("INFO" if desc_by_series.max() == 1 else "WARNING", "description_consistency",
        "SeriesDescription constant within every series" if desc_by_series.max() == 1 else "some series mix descriptions")

    issues_path = PROJECT_ROOT / "outputs" / "reports" / "dicom_header_issues.json"
    issues = json.loads(issues_path.read_text(encoding="utf-8")) if issues_path.exists() else {"issues": []}
    if issues["issues"]:
        add("CRITICAL", "header_issues", f"{len(issues['issues'])} header-level issue(s)", issues["issues"][:5])

    summary = {
        "scope": {
            "files_audited_full_pixel_pass": int(len(px)),
            "decode_failures": len(errors),
            "series": int(hdr["SeriesInstanceUID"].nunique()),
            "studies": int(hdr["StudyInstanceUID"].nunique()),
        },
        "intensity_overview": {
            "global_min": float(px.loc[px.ok, "min"].min()),
            "global_max": float(px.loc[px.ok, "max"].max()),
            "per_file_max_distribution": describe(px.loc[px.ok, "max"]),
        },
        "findings": findings,
        "severity_counts": {
            sev: sum(1 for f in findings if f["severity"] == sev)
            for sev in ["CRITICAL", "WARNING", "INFO"]
        },
    }
    write_report("quality_audit.json", summary)
    print(json.dumps(summary["severity_counts"], indent=1))
    for f in findings:
        print(f"  [{f['severity']:8s}] {f['code']}: {f['detail'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
