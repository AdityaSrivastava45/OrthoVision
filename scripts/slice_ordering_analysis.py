"""PHASE 6 - Slice ordering: InstanceNumber vs physical geometry.

For every on-disk series, compare InstanceNumber rank order against the
anatomical order given by projecting ImagePositionPatient onto the slice
normal (IOP_row x IOP_col). Reports Spearman correlation, monotonicity,
direction, InstanceNumber uniqueness/contiguity, and spacing uniformity.
Writes slice_ordering_analysis.json.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import PROJECT_ROOT, describe, write_report

sys.path.insert(0, str(PROJECT_ROOT / "src"))


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

    rows = []
    for series_uid, grp in hdr.groupby("SeriesInstanceUID"):
        g = grp.copy()
        inst = pd.to_numeric(g["InstanceNumber"], errors="coerce")
        normals = []
        positions = []
        for _, r in g.iterrows():
            iop = parse_vec(r["ImageOrientationPatient"])
            ipp = parse_vec(r["ImagePositionPatient"])
            if iop is None or len(iop) < 6 or ipp is None:
                continue
            n = np.cross(iop[:3], iop[3:6])
            norm = float(np.linalg.norm(n))
            if norm < 1e-8:
                continue
            n /= norm
            normals.append(n)
            positions.append(float(np.dot(ipp, n)))
        entry = {
            "series_uid_tail": series_uid[-12:],
            "n_slices": int(len(g)),
            "instance_number_unique": bool(inst.nunique() == len(inst.dropna())),
            "instance_number_missing": int(inst.isna().sum()),
            "instance_numbers_contiguous_from_1": None,
            "n_spatial": len(positions),
        }
        inst_vals = sorted(int(v) for v in inst.dropna())
        if inst_vals:
            entry["instance_numbers_contiguous_from_1"] = bool(
                inst_vals == list(range(1, len(inst_vals) + 1))
            )
            entry["instance_min"] = int(min(inst_vals))
            entry["instance_max"] = int(max(inst_vals))
        if len(positions) == len(g) and len(positions) >= 2 and normals:
            ref = normals[0]
            consistent = all(float(np.dot(ref, n)) > 0.99 for n in normals)
            pos = np.asarray(positions)
            inst_series = pd.Series(inst.values[: len(pos)], dtype="float64")
            rho = float(inst_series.rank().corr(pd.Series(pos).rank()))
            d = np.abs(np.diff(np.sort(pos)))
            entry.update(
                {
                    "normals_consistent_within_series": bool(consistent),
                    "position_span_mm": round(float(pos.max() - pos.min()), 4),
                    "median_gap_mm": round(float(np.median(d)), 4),
                    "gap_uniformity_cv": (
                        round(float(np.std(d) / np.mean(d)), 4) if np.mean(d) > 0 else None
                    ),
                    "duplicate_positions_within_1um": int((np.diff(np.sort(pos)) < 1e-3).sum()),
                    "spearman_instnum_vs_position": None if np.isnan(rho) else round(float(rho), 4),
                    "direction": (
                        None
                        if np.isnan(rho)
                        else ("increasing_instance_is_increasing_position" if rho > 0 else "decreasing")
                    ),
                }
            )
        rows.append(entry)

    df = pd.DataFrame(rows)
    with_rho = df[df["spearman_instnum_vs_position"].notna()]

    report = {
        "method": "project IPP onto normal=IOP_row x IOP_col (LPS); compare ranks vs InstanceNumber",
        "n_series": int(len(df)),
        "series_fully_spatial": int((df["n_spatial"] == df["n_slices"]).sum()),
        "instance_number_unique_rate": float(df["instance_number_unique"].mean()),
        "instance_number_missing_total": int(df["instance_number_missing"].sum()),
        "contiguous_from_1_rate": (
            float((df["instance_numbers_contiguous_from_1"] == True).mean())
            if df["instance_numbers_contiguous_from_1"].notna().any()
            else None
        ),
        "spearman_distribution": describe(with_rho["spearman_instnum_vs_position"]),
        "abs_spearman_ge_0_999_rate": float((with_rho["spearman_instnum_vs_position"].abs() >= 0.999).mean()),
        "direction_counts": {str(k): int(v) for k, v in with_rho["direction"].value_counts().items()},
        "gap_uniformity_cv_max": float(with_rho["gap_uniformity_cv"].max()),
        "series_with_duplicate_positions": int((df["duplicate_positions_within_1um"].fillna(0) > 0).sum()),
        "per_series": df.to_dict(orient="records"),
    }

    write_report("slice_ordering_analysis.json", report)
    print(f"series={len(df)} fully_spatial={report['series_fully_spatial']} "
          f"|rho|>=0.999 rate={report['abs_spearman_ge_0_999_rate']}")
    print("direction counts:", report["direction_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
