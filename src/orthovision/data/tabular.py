"""CSV layer for the RSNA knee MRI tables.

Loads train/test/series/submission CSVs and computes label statistics.
Missing labels are preserved as NaN - never imputed here.

Schema observed 2026-08 (see docs/DATASET_INTELLIGENCE.md):
    train.csv            StudyInstanceUID, Report, <12 label columns>
    train_series.csv     StudyInstanceUID, SeriesInstanceUID,
                         Fluid_Sensitive, Fat_Suppression, Anatomical_Plane
    test.csv             StudyInstanceUID
    sample_submission    StudyInstanceUID + 12 float columns
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABELS: tuple[str, ...] = (
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

SERIES_COLUMNS: tuple[str, ...] = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
)

PLANES: tuple[str, ...] = ("Axial", "Coronal", "Sagittal")

KNOWN_PLANES: frozenset[str] = frozenset(PLANES)


class TabularSchemaError(ValueError):
    """Raised when a CSV is missing expected columns. Fail loudly."""


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise TabularSchemaError(f"{name}: missing columns {missing}; found {list(df.columns)}")


def load_train(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, ("StudyInstanceUID", *LABELS), "train")
    if "Report" not in df.columns:
        raise TabularSchemaError("train: 'Report' column absent; reports are train-time privileged data")
    return df


def load_series(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, SERIES_COLUMNS, "series")
    unknown = set(df["Anatomical_Plane"].dropna().unique()) - KNOWN_PLANES
    if unknown:
        raise TabularSchemaError(f"series: unexpected Anatomical_Plane values {sorted(unknown)}")
    return df


def load_test(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, ("StudyInstanceUID",), "test")
    return df


def load_sample_submission(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, ("StudyInstanceUID", *LABELS), "sample_submission")
    return df


@dataclass(frozen=True)
class LabelStats:
    label: str
    total: int
    labeled: int
    positive: int
    negative: int
    missing: int
    prevalence_labeled: float
    positive_values: tuple[float, ...]

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "total": self.total,
            "labeled": self.labeled,
            "positive": self.positive,
            "negative": self.negative,
            "missing": self.missing,
            "prevalence_labeled": self.prevalence_labeled,
            "positive_values": list(self.positive_values),
        }


def label_statistics(train_df: pd.DataFrame, labels: tuple[str, ...] = LABELS) -> list[LabelStats]:
    stats: list[LabelStats] = []
    for label in labels:
        col = train_df[label]
        labeled = col.dropna()
        pos = labeled[labeled > 0]
        neg = labeled[labeled == 0]
        prevalence = float(len(pos) / len(labeled)) if len(labeled) else float("nan")
        stats.append(
            LabelStats(
                label=label,
                total=int(len(col)),
                labeled=int(len(labeled)),
                positive=int(len(pos)),
                negative=int(len(neg)),
                missing=int(len(col) - len(labeled)),
                prevalence_labeled=prevalence,
                positive_values=tuple(sorted(float(v) for v in pos.unique())),
            )
        )
    return stats


def cooccurrence_matrix(
    train_df: pd.DataFrame,
    labels: tuple[str, ...] = LABELS,
    *,
    binarize_threshold: float = 0.0,
) -> pd.DataFrame:
    """Joint positive counts P(A=1 and B=1) among studies labeled for both."""
    out = pd.DataFrame(0, index=list(labels), columns=list(labels), dtype=int)
    cols = {label: train_df[label] for label in labels}
    for i, a in enumerate(labels):
        col_a = cols[a]
        pos_a = col_a > binarize_threshold
        notna_a = col_a.notna()
        out.loc[a, a] = int((pos_a & notna_a).sum())
        for b in labels[i + 1 :]:
            col_b = cols[b]
            joint = int((pos_a & (col_b > binarize_threshold) & notna_a & col_b.notna()).sum())
            out.loc[a, b] = joint
            out.loc[b, a] = joint
    return out


def series_per_study(series_df: pd.DataFrame) -> pd.Series:
    counts = series_df.groupby("StudyInstanceUID")["SeriesInstanceUID"].nunique()
    return counts.sort_index()
