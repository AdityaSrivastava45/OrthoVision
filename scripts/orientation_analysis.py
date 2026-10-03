"""PHASE 5 - CSV Anatomical_Plane vs DICOM-derived orientation.

For every on-disk series: derive the slice normal from ImageOrientationPatient
(row x column cosines, LPS frame), classify plane by dominant axis with
configurable alignment/separation thresholds (defaults from configs/data.yaml),
and compare against the CSV label. Writes orientation_analysis.json.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import PROJECT_ROOT, fmt_counts, write_report

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from orthovision.data.tabular import load_series


AXES = {"sagittal": 0, "coronal": 1, "axial": 2}
CSV_TO_SPATIAL = {"Sagittal": "sagittal", "Coronal": "coronal", "Axial": "axial"}


def parse_vec(text):
    if not isinstance(text, str):
        return None
    try:
        return np.array([float(x) for x in text.split(";")], dtype=np.float64)
    except ValueError:
        return None


def main() -> int:
    hdr = pd.read_csv(
        PROJECT_ROOT / "outputs" / "reports" / "dicom_headers.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    ts = load_series(PROJECT_ROOT / "data" / "raw" / "train_series.csv")
    csv_plane = ts.set_index("SeriesInstanceUID")["Anatomical_Plane"]

    min_alignment = 0.8
    min_separation = 0.2

    rows = []
    for series_uid, grp in hdr.groupby("SeriesInstanceUID"):
        normals = []
        ipps = []
        for _, r in grp.iterrows():
            iop = parse_vec(r["ImageOrientationPatient"])
            ipp = parse_vec(r["ImagePositionPatient"])
            if iop is None or len(iop) < 6:
                continue
            n = np.cross(iop[:3], iop[3:6])
            norm = np.linalg.norm(n)
            if norm < 1e-8:
                continue
            normals.append(n / norm)
            if ipp is not None and len(ipp) == 3:
                ipps.append(ipp)
        entry = {
            "series_uid_tail": series_uid[-12:],
            "csv_plane": csv_plane.get(series_uid),
            "n_slices": int(len(grp)),
            "n_usable_normals": len(normals),
        }
        if normals:
            stack = np.vstack(normals)
            spread = float(np.max(np.linalg.norm(stack - stack.mean(axis=0), axis=1)))
            ref = stack.mean(axis=0)
            ref = ref / np.linalg.norm(ref)
            abs_n = np.abs(ref)
            order = np.argsort(abs_n)[::-1]
            best, second = int(order[0]), int(order[1])
            align = {name: float(abs_n[i]) for name, i in AXES.items()}
            entry.update(
                {
                    "normal_lps": [round(float(x), 6) for x in ref],
                    "alignment": {k: round(v, 6) for k, v in align.items()},
                    "within_series_normal_spread": round(spread, 8),
                    "dominant_axis": ["sagittal", "coronal", "axial"][best],
                    "alignment_of_best": round(float(abs_n[best]), 6),
                    "separation_best_minus_second": round(float(abs_n[best] - abs_n[second]), 6),
                }
            )
            if abs_n[best] < min_alignment or (abs_n[best] - abs_n[second]) < min_separation:
                entry["derived_plane"] = "unknown"
                entry["reason"] = "oblique or ambiguous normal"
            else:
                entry["derived_plane"] = ["sagittal", "coronal", "axial"][best]
        else:
            entry["derived_plane"] = None
            entry["reason"] = "no usable IOP"
        rows.append(entry)

    df = pd.DataFrame(rows)

    def expected_spatial(csv_plane):
        return CSV_TO_SPATIAL.get(csv_plane)

    comparable = df[df["derived_plane"].notna() & df["csv_plane"].notna()].copy()
    comparable["expected"] = comparable["csv_plane"].map(expected_spatial)
    comparable["agree"] = comparable["derived_plane"] == comparable["expected"]

    report = {
        "method": "normal = normalize(IOP_row x IOP_col) in LPS; X->sagittal Y->coronal Z->axial; "
        f"plane requires max|axis|>={min_alignment} and margin>={min_separation}",
        "thresholds": {"min_alignment": min_alignment, "min_axis_separation": min_separation},
        "n_series_on_disk": int(len(df)),
        "series_with_derivable_plane": int(df["derived_plane"].notna().sum()),
        "agreement_rate_overall": float(comparable["agree"].mean()),
        "agreement_by_csv_plane": {
            plane: {
                "n": int((comparable["csv_plane"] == plane).sum()),
                "agree": int(comparable.loc[comparable["csv_plane"] == plane, "agree"].sum()),
            }
            for plane in ["Sagittal", "Coronal", "Axial"]
        },
        "confusions": [
            {"csv_plane": r["csv_plane"], "derived_plane": r["derived_plane"], "series_tail": r["series_uid_tail"]}
            for _, r in comparable[~comparable["agree"]].iterrows()
        ],
        "unknown_plane_series": [
            {"series_tail": r["series_uid_tail"], "csv_plane": r["csv_plane"], "alignment": r.get("alignment")}
            for _, r in df[df["derived_plane"] == "unknown"].iterrows()
        ],
        "per_series": json_ready(df),
        "iop_constant_within_series_rate": float(
            (df["within_series_normal_spread"].dropna() < 1e-6).mean()
        ) if df["within_series_normal_spread"].notna().any() else None,
    }

    sign_info = []
    for series_uid, grp in hdr.groupby("SeriesInstanceUID"):
        iop0 = parse_vec(grp.iloc[0]["ImageOrientationPatient"])
        if iop0 is None:
            continue
        n = np.cross(iop0[:3], iop0[3:6])
        n /= np.linalg.norm(n)
        sign_info.append({"series_tail": series_uid[-12:], "sign_x": float(np.sign(n[0])), "sign_y": float(np.sign(n[1])), "sign_z": float(np.sign(n[2]))})
    sdf = pd.DataFrame(sign_info)
    report["normal_sign_conventions_observed"] = (
        sdf.groupby(["sign_x", "sign_y", "sign_z"]).size().reset_index(name="n_series").to_dict(orient="records")
    )

    write_report("orientation_analysis.json", report)
    print(f"series={len(df)} agreement={report['agreement_rate_overall']:.4f} "
          f"unknown={len(report['unknown_plane_series'])}")
    return 0


def json_ready(df: pd.DataFrame):
    def clean(v):
        if isinstance(v, (list, dict, tuple)) or v is None:
            return v
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return v

    return [{k: clean(v) for k, v in r.items()} for _, r in df.iterrows()]


if __name__ == "__main__":
    raise SystemExit(main())
