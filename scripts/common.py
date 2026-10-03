"""Shared helpers for Stage-1 analysis scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

LABEL_COLUMNS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)


def reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def write_report(name: str, payload: dict) -> Path:
    out = reports_dir() / name
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"[report] {out}")
    return out


def _json_default(obj):
    import numpy as np
    from pathlib import Path as _Path

    if isinstance(obj, (_Path, np.integer)):
        return str(obj) if isinstance(obj, _Path) else int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def fmt_counts(series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def describe(values) -> dict:
    import numpy as np

    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }
