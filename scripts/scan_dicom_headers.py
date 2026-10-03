"""Full DICOM header scan of the available subset under data/raw.

Writes one row per file to outputs/reports/dicom_headers.csv (headers only,
stop_before_pixels=True; pixel data is never read here) plus issues list.
Deterministic: sorted file order, single-threaded by default.

Usage:
    python scripts/scan_dicom_headers.py [--workers N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from common import PROJECT_ROOT, reports_dir

from orthovision.data.discovery import discover_dicom_headers


HEADER_FIELDS = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SeriesDescription",
    "SeriesNumber",
    "Modality",
    "BodyPartExamined",
    "ProtocolName",
    "SequenceName",
    "PhotometricInterpretation",
    "PatientID",
    "StudyDate",
    "StudyDescription",
    "Manufacturer",
    "ManufacturerModelName",
    "Laterality",
    "PatientPosition",
    "TransferSyntaxUID",
    "Rows",
    "Columns",
    "InstanceNumber",
    "NumberOfFrames",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "SamplesPerPixel",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "RescaleSlope",
    "RescaleIntercept",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    root = PROJECT_ROOT / "data" / "raw"
    if not root.exists():
        print(f"FATAL: dataset root missing: {root}", file=sys.stderr)
        return 2

    headers, issues, counts = discover_dicom_headers(root, num_workers=args.workers)
    rows = []
    for h in headers:
        m = h.metadata
        row = {"path": str(h.path.relative_to(root))}
        for field in HEADER_FIELDS:
            value = m.get(field)
            if isinstance(value, tuple):
                value = ";".join(f"{v:.8g}" for v in value)
            row[field] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = reports_dir() / "dicom_headers.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    issues_path = reports_dir() / "dicom_header_issues.json"
    issues_payload = {
        "counts": counts,
        "issues": [
            {"path": i.path, "stage": i.stage, "message": i.message, "kind": i.kind}
            for i in issues
        ],
    }
    import json

    with issues_path.open("w", encoding="utf-8") as f:
        json.dump(issues_payload, f, indent=2)

    print(f"files_seen={counts['files_seen']} dicom_valid={counts['dicom_valid']} "
          f"unreadable={counts['dicom_unreadable']} not_dicom={counts['not_dicom']}")
    print(f"wrote {out_csv} ({len(df)} rows) and {issues_path} ({len(issues)} issues)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
