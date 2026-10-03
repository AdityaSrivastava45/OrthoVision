"""PHASE 4 - DICOM metadata + pixel characteristics.

Uses the cached header scan for population statistics, then decodes a small,
deterministic pixel sample: for every on-disk series, first/middle/last slice
(ordered by projected position when available, else InstanceNumber).
Writes outputs/reports/dicom_pixel_analysis.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import PROJECT_ROOT, describe, write_report

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from orthovision.data.pixels import PixelDecodeError, decode_path
from orthovision.data.tabular import load_series


def order_key(row):
    iop = row.get("ImageOrientationPatient")
    ipp = row.get("ImagePositionPatient")
    if isinstance(iop, str) and isinstance(ipp, str):
        try:
            v = [float(x) for x in iop.split(";")]
            p = [float(x) for x in ipp.split(";")]
            r = np.array(v[:3])
            c = np.array(v[3:6])
            n = np.cross(r, c)
            return float(np.dot(p, n))
        except Exception:
            pass
    inst = row["InstanceNumber"]
    return float(inst) if pd.notna(inst) else 0.0


def main() -> int:
    raw_root = PROJECT_ROOT / "data" / "raw"
    hdr = pd.read_csv(
        PROJECT_ROOT / "outputs" / "reports" / "dicom_headers.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    ts = load_series(raw_root / "train_series.csv")
    meta = ts.set_index("SeriesInstanceUID")[["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]]
    hdr = hdr.join(meta, on="SeriesInstanceUID")

    report: dict = {"population_header_stats": {}}

    pop = report["population_header_stats"]
    pop["iop_present"] = int(hdr["ImageOrientationPatient"].notna().sum())
    pop["ipp_present"] = int(hdr["ImagePositionPatient"].notna().sum())
    pop["instance_number_present"] = int(hdr["InstanceNumber"].notna().sum())
    pop["n_files"] = int(len(hdr))
    pop["pixel_spacing_row_mm"] = describe(hdr["PixelSpacing"].dropna().map(lambda s: float(s.split(";")[0])))
    pop["pixel_spacing_col_mm"] = describe(hdr["PixelSpacing"].dropna().map(lambda s: float(s.split(";")[-1])))
    pop["slice_thickness_mm"] = describe(hdr["SliceThickness"].dropna().astype(float))
    pop["spacing_between_slices_mm"] = describe(hdr["SpacingBetweenSlices"].dropna().astype(float))
    pop["rescale_slope_values"] = {
        "present": int(hdr["RescaleSlope"].notna().sum()),
        "equal_1": int((hdr["RescaleSlope"].astype(float) == 1.0).sum()),
        "non_trivial_examples": sorted(
            {round(float(v), 3) for v in hdr["RescaleSlope"].dropna().unique() if abs(float(v) - 1.0) > 1e-6}
        )[:10],
    }
    pop["series_descriptions_top15"] = {
        str(k): int(v) for k, v in hdr["SeriesDescription"].value_counts().head(15).items()
    }

    sampled = []
    for series_uid, grp in hdr.groupby("SeriesInstanceUID"):
        g = grp.copy()
        g["_key"] = g.apply(order_key, axis=1)
        g = g.sort_values("_key")
        picks = {0: "first", len(g) // 2: "middle", len(g) - 1: "last"}
        for idx, tag in picks.items():
            row = g.iloc[idx]
            path = raw_root / row["path"]
            entry = {
                "path": row["path"],
                "series_uid_tail": series_uid[-12:],
                "plane_csv": row.get("Anatomical_Plane"),
                "fluid_sensitive": None if pd.isna(row.get("Fluid_Sensitive")) else int(row["Fluid_Sensitive"]),
                "fat_suppression": None if pd.isna(row.get("Fat_Suppression")) else int(row["Fat_Suppression"]),
                "position_in_series": tag,
                "rows": int(row["Rows"]),
                "columns": int(row["Columns"]),
            }
            try:
                payload = decode_path(path)
                for name, arr in (("raw", payload.raw), ("rescaled", payload.rescaled)):
                    finite = arr[np.isfinite(arr)]
                    entry[f"{name}_dtype"] = str(arr.dtype)
                    entry[f"{name}_min"] = float(np.min(finite))
                    entry[f"{name}_max"] = float(np.max(finite))
                    entry[f"{name}_mean"] = float(np.mean(finite))
                    entry[f"{name}_std"] = float(np.std(finite))
                    entry[f"{name}_p1"] = float(np.percentile(finite, 1))
                    entry[f"{name}_p99"] = float(np.percentile(finite, 99))
                entry["slope"] = payload.rescale_slope
                entry["intercept"] = payload.rescale_intercept
                entry["decode_error"] = None
            except PixelDecodeError as exc:
                entry["decode_error"] = str(exc)
            sampled.append(entry)

    sdf = pd.DataFrame(sampled)
    errors = sdf[sdf["decode_error"].notna()]
    report["pixel_sample"] = {
        "strategy": "first/middle/last of every on-disk series ordered by IPP.normal (fallback InstanceNumber)",
        "n_sampled": int(len(sdf)),
        "n_decode_errors": int(len(errors)),
        "decode_errors": errors["decode_error"].tolist()[:5],
    }

    ok = sdf[sdf["decode_error"].isna()]
    agg_rows = []
    for (plane, fluid), grp in ok.groupby(["plane_csv", "fluid_sensitive"], dropna=False):
        agg_rows.append(
            {
                "plane_csv": plane,
                "fluid_sensitive": fluid,
                "n": int(len(grp)),
                "raw_min_min": float(grp["raw_min"].min()),
                "raw_max_max": float(grp["raw_max"].max()),
                "raw_mean_avg": float(grp["raw_mean"].mean()),
                "raw_p1_min": float(grp["raw_p1"].min()),
                "raw_p99_max": float(grp["raw_p99"].max()),
            }
        )
    report["intensity_by_plane_fluid"] = agg_rows

    report["raw_dtype_values"] = {str(k): int(v) for k, v in ok["raw_dtype"].value_counts().items()}
    report["raw_global_range"] = {
        "min_of_min": float(ok["raw_min"].min()),
        "max_of_max": float(ok["raw_max"].max()),
        "note": "unsigned 12-bit stored; expect [0, 4095]",
    }
    rescaled_differs = ok[(ok["slope"].fillna(1.0) != 1.0) | (ok["intercept"].fillna(0.0) != 0.0)]
    report["rescale_effect_on_sample"] = {
        "files_where_rescale_changes_values": int(len(rescaled_differs)),
        "examples_raw_vs_rescaled": [
            {
                "path": p,
                "slope": float(s),
                "raw_max": float(m),
                "rescaled_max": float(rm),
            }
            for p, s, m, rm in zip(
                rescaled_differs["path"].head(5),
                rescaled_differs["slope"].head(5),
                rescaled_differs["raw_max"].head(5),
                rescaled_differs["rescaled_max"].head(5),
            )
        ],
    }

    write_report("dicom_pixel_analysis.json", report)
    print(f"sampled={len(sdf)} decode_errors={len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
