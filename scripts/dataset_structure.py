"""PHASE 2 - Dataset structure analysis.

Combines the four CSVs with the cached DICOM header scan and the filesystem.
Writes outputs/reports/dataset_structure.json.

Observed on-disk layout (verified programmatically):
    train_series/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
The script detects which UID keys the directories so it stays correct if the
layout changes to <StudyInstanceUID>/... later.

Coverage note: train.csv/train_series.csv describe the FULL corpus;
the on-disk DICOM subset is smaller. Scopes are reported separately.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

import pandas as pd

from common import PROJECT_ROOT, describe, fmt_counts, write_report

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from orthovision.data.tabular import LABELS, load_sample_submission, load_series, load_test, load_train, series_per_study


def norm_parts(rel_path: str) -> tuple[str, ...]:
    return PurePosixPath(rel_path.replace("\\", "/")).parts


def main() -> int:
    raw = PROJECT_ROOT / "data" / "raw"
    train = load_train(raw / "train.csv")
    train_series = load_series(raw / "train_series.csv")
    test = load_test(raw / "test.csv")
    test_series = load_series(raw / "test_series.csv")
    sub = load_sample_submission(raw / "sample_submission.csv")

    headers_path = PROJECT_ROOT / "outputs" / "reports" / "dicom_headers.csv"
    if not headers_path.exists():
        print("FATAL: run scripts/scan_dicom_headers.py first", file=sys.stderr)
        return 2
    hdr = pd.read_csv(headers_path, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    hdr["dir1"] = hdr["path"].map(lambda p: norm_parts(p)[1] if len(norm_parts(p)) > 1 else None)

    report: dict = {"coverage": {}}

    n_studies_csv = int(train["StudyInstanceUID"].nunique())
    n_series_csv = int(train_series["SeriesInstanceUID"].nunique())
    sps = series_per_study(train_series)

    report["coverage"]["train_csv"] = {
        "rows": int(len(train)),
        "unique_studies": n_studies_csv,
        "duplicate_study_rows": int(len(train) - n_studies_csv),
    }
    report["coverage"]["train_series_csv"] = {
        "rows": int(len(train_series)),
        "unique_series": n_series_csv,
        "studies_represented": int(train_series["StudyInstanceUID"].nunique()),
    }
    report["coverage"]["test_csv"] = {
        "rows": int(len(test)),
        "unique_studies": int(test["StudyInstanceUID"].nunique()),
    }
    report["coverage"]["test_series_csv"] = {
        "rows": int(len(test_series)),
        "unique_series": int(test_series["SeriesInstanceUID"].nunique()),
        "studies_represented": int(test_series["StudyInstanceUID"].nunique()),
    }
    report["coverage"]["sample_submission"] = {
        "rows": int(len(sub)),
        "values_all_zero_point_five": bool((sub[list(LABELS)] == 0.5).all().all()),
        "study_ids_match_test_csv": bool(set(sub["StudyInstanceUID"]) == set(test["StudyInstanceUID"])),
    }

    report["series_per_study"] = {
        "train_csv_full_corpus": describe(sps),
        "distribution": {str(k): int(v) for k, v in sps.value_counts().sort_index().items()},
    }

    disk_dir_names = sorted({p.name for p in (raw / "train_series").iterdir() if p.is_dir()})
    csv_study_ids = set(train["StudyInstanceUID"])
    csv_series_ids = set(train_series["SeriesInstanceUID"])
    dirs_are_series = sum(d in csv_series_ids for d in disk_dir_names)
    dirs_are_studies = sum(d in csv_study_ids for d in disk_dir_names)
    if dirs_are_series >= dirs_are_studies:
        layout = "train_series/<SeriesInstanceUID>/<SOPInstanceUID>.dcm"
        keyed_by = "series"
    else:
        layout = "train_series/<StudyInstanceUID>/..."
        keyed_by = "study"

    disk_study_ids = set(hdr["StudyInstanceUID"].dropna())
    disk_series_ids = set(hdr["SeriesInstanceUID"].dropna())
    slices_per_series = hdr.groupby("SeriesInstanceUID")["SOPInstanceUID"].size()

    report["coverage"]["disk_subset"] = {
        "layout_detected": layout,
        "top_level_dir_count": len(disk_dir_names),
        "dirs_keyed_by": keyed_by,
        "dicom_files_parsed": int(len(hdr)),
        "unique_series_on_disk": len(disk_series_ids),
        "unique_studies_on_disk": len(disk_study_ids),
        "studies_on_disk": sorted(disk_study_ids),
    }

    report["slices_per_series"] = {
        "disk_subset": describe(slices_per_series),
        "min": int(slices_per_series.min()),
        "max": int(slices_per_series.max()),
        "distribution": {int(k): int(v) for k, v in slices_per_series.value_counts().sort_index().items()},
    }

    report["plane_distribution"] = {
        "train_series_csv_full": fmt_counts(train_series["Anatomical_Plane"]),
        "test_series_csv": fmt_counts(test_series["Anatomical_Plane"]),
        "disk_subset": fmt_counts(
            train_series.set_index("SeriesInstanceUID").loc[sorted(disk_series_ids & csv_series_ids), "Anatomical_Plane"]
        )
        if disk_series_ids & csv_series_ids
        else {},
    }
    report["fluid_sensitive"] = {
        "train_series_csv_full": fmt_counts(train_series["Fluid_Sensitive"]),
        "by_plane_full_corpus": {
            plane: fmt_counts(train_series.loc[train_series["Anatomical_Plane"] == plane, "Fluid_Sensitive"])
            for plane in ["Sagittal", "Coronal", "Axial"]
        },
    }
    report["fat_suppression"] = {
        "train_series_csv_full": fmt_counts(train_series["Fat_Suppression"]),
        "by_plane_full_corpus": {
            plane: fmt_counts(train_series.loc[train_series["Anatomical_Plane"] == plane, "Fat_Suppression"])
            for plane in ["Sagittal", "Coronal", "Axial"]
        },
    }
    combo = (
        train_series.groupby(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"])
        .size()
        .reset_index(name="n")
    )
    report["plane_fluid_fat_combinations"] = combo.to_dict(orient="records")

    report["image_dimensions"] = {
        "rows_pixels": {str(k): int(v) for k, v in hdr["Rows"].value_counts().items()},
        "columns_pixels": {str(k): int(v) for k, v in hdr["Columns"].value_counts().items()},
        "non_square_or_mixed_note": str(
            hdr.loc[hdr["Rows"] != hdr["Columns"], ["Rows", "Columns"]].value_counts().to_dict()
        ),
    }

    report["missing_values"] = {
        "train_csv_non_label_columns": {c: int(n) for c, n in train[["StudyInstanceUID", "Report"]].isna().sum().items() if n > 0},
        "train_labels_missing_total_across_12": int(train[[l for l in train.columns if l not in ("StudyInstanceUID", "Report")]].isna().sum().sum()),
        "train_series_csv": {c: int(n) for c, n in train_series.isna().sum().items() if n > 0},
        "test_series_csv": {c: int(n) for c, n in test_series.isna().sum().items() if n > 0},
    }

    label_cols = [c for c in train.columns if c not in ("StudyInstanceUID", "Report")]
    report["duplicates"] = {
        "train_duplicate_studyinstanceuid": int(train["StudyInstanceUID"].duplicated().sum()),
        "train_series_duplicate_series_uid": int(train_series["SeriesInstanceUID"].duplicated().sum()),
        "train_series_duplicate_study_series_pair": int(
            train_series.duplicated(subset=["StudyInstanceUID", "SeriesInstanceUID"]).sum()
        ),
        "dicom_duplicate_sop_uid_within_scan": int(hdr["SOPInstanceUID"].duplicated().sum()),
        "label_cols_checked": label_cols,
    }

    orphan_disk_series = sorted(disk_series_ids - csv_series_ids)
    csv_only_series = sorted(csv_series_ids - disk_series_ids)
    orphan_disk_studies = sorted(disk_study_ids - csv_study_ids)

    folder_uid_agreement = float((hdr["dir1"] == hdr["SeriesInstanceUID"]).mean()) if keyed_by == "series" else None
    ser_to_study = dict(zip(train_series["SeriesInstanceUID"], train_series["StudyInstanceUID"]))
    mapping_ok = all(
        ser_to_study.get(s) == u
        for s, u in zip(hdr["SeriesInstanceUID"], hdr["StudyInstanceUID"])
        if s in ser_to_study
    )

    report["orphans_and_consistency"] = {
        "disk_series_not_in_train_series_csv": len(orphan_disk_series),
        "examples": orphan_disk_series[:5],
        "csv_series_without_disk_files": len(csv_only_series),
        "disk_studies_not_in_train_csv": len(orphan_disk_studies),
        "csv_studies_without_disk_data": len(csv_study_ids - disk_study_ids),
        "files_missing_study_or_series_uid": int(
            hdr["SeriesInstanceUID"].isna().sum() + hdr["StudyInstanceUID"].isna().sum()
        ),
        "folder_name_equals_header_series_uid_rate": folder_uid_agreement,
        "csv_series_to_study_mapping_matches_dicom_headers": bool(mapping_ok),
        "train_test_study_overlap": len(set(test["StudyInstanceUID"]) & csv_study_ids),
        "train_test_series_overlap": len(set(test_series["SeriesInstanceUID"]) & csv_series_ids),
    }

    csv_counts = train_series.groupby("StudyInstanceUID")["SeriesInstanceUID"].nunique()
    disk_counts = hdr.groupby("StudyInstanceUID")["SeriesInstanceUID"].nunique()
    cmp_df = pd.DataFrame({"csv": csv_counts}).join(disk_counts.rename("disk"), how="inner")
    mismatched = cmp_df[cmp_df["csv"] != cmp_df["disk"]]
    report["disk_completeness_vs_csv"] = {
        "studies_with_partial_disk_data": int(len(mismatched)),
        "detail": {k[-12:]: {"csv_series": int(v["csv"]), "disk_series": int(v["disk"])} for k, v in mismatched.iterrows()},
    }

    write_report("dataset_structure.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
