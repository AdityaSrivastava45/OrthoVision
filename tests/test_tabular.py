from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orthovision.data.tabular import (
    LABELS,
    TabularSchemaError,
    cooccurrence_matrix,
    label_statistics,
    load_sample_submission,
    load_series,
    load_test,
    load_train,
    series_per_study,
)


LABEL_COLS = list(LABELS)


def write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def sample_train() -> pd.DataFrame:
    rows = [
        {"StudyInstanceUID": f"study-{i}", "Report": f"report {i}", **{c: np.nan for c in LABEL_COLS}}
        for i in range(5)
    ]
    rows[0].update({"ACL": 1.0, "Effusion": 1.0, "MCL": 0.0})
    rows[1].update({"ACL": 0.0, "MCL": 0.0})
    rows[2].update({"ACL": 1.0})
    df = pd.DataFrame(rows)
    return df[list(df.columns)]


@pytest.fixture
def train_csv(tmp_path: Path) -> Path:
    return write_csv(tmp_path / "train.csv", sample_train())


def test_load_train_roundtrip(train_csv: Path):
    df = load_train(train_csv)
    assert len(df) == 5
    assert set(LABEL_COLS).issubset(df.columns)


def test_load_train_missing_column_raises(tmp_path: Path):
    bad = sample_train().drop(columns=["Baker's"])
    p = write_csv(tmp_path / "bad.csv", bad)
    with pytest.raises(TabularSchemaError):
        load_train(p)


def test_load_train_missing_report_raises(tmp_path: Path):
    bad = sample_train().drop(columns=["Report"])
    p = write_csv(tmp_path / "noreport.csv", bad)
    with pytest.raises(TabularSchemaError):
        load_train(p)


def test_label_statistics_counts_and_prevalence():
    stats = {s.label: s for s in label_statistics(sample_train())}
    acl = stats["ACL"]
    assert (acl.labeled, acl.positive, acl.negative) == (3, 2, 1)
    assert acl.missing == 2
    assert acl.prevalence_labeled == pytest.approx(2 / 3)
    mcl = stats["MCL"]
    assert (mcl.labeled, mcl.positive) == (2, 0)
    assert mcl.prevalence_labeled == 0.0
    fracture = stats["Fracture"]
    assert fracture.labeled == 0 and fracture.missing == 5
    assert np.isnan(fracture.prevalence_labeled)


def test_cooccurrence_matrix_matches_hand_count():
    joint = cooccurrence_matrix(sample_train())
    assert joint.loc["ACL", "ACL"] == 2
    assert joint.loc["ACL", "MCL"] == 0
    assert joint.loc["ACL", "Effusion"] == 1
    assert joint.loc["Effusion", "ACL"] == 1
    assert joint.loc["Fracture", "ACL"] == 0


def test_series_per_study_counts(tmp_path: Path):
    rows = [
        {"StudyInstanceUID": "s1", "SeriesInstanceUID": "a", "Fluid_Sensitive": 1, "Fat_Suppression": 1, "Anatomical_Plane": "Sagittal"},
        {"StudyInstanceUID": "s1", "SeriesInstanceUID": "b", "Fluid_Sensitive": 0, "Fat_Suppression": 0, "Anatomical_Plane": "Axial"},
        {"StudyInstanceUID": "s2", "SeriesInstanceUID": "c", "Fluid_Sensitive": 1, "Fat_Suppression": 1, "Anatomical_Plane": "Coronal"},
    ]
    p = write_csv(tmp_path / "series.csv", pd.DataFrame(rows))
    df = load_series(p)
    counts = series_per_study(df)
    assert counts["s1"] == 2 and counts["s2"] == 1


def test_load_series_rejects_unknown_plane(tmp_path: Path):
    rows = [
        {"StudyInstanceUID": "s1", "SeriesInstanceUID": "a", "Fluid_Sensitive": 1, "Fat_Suppression": 1, "Anatomical_Plane": "Oblique45"}
    ]
    p = write_csv(tmp_path / "series.csv", pd.DataFrame(rows))
    with pytest.raises(TabularSchemaError):
        load_series(p)


def test_load_test_and_submission(tmp_path: Path):
    test_p = write_csv(tmp_path / "test.csv", pd.DataFrame({"StudyInstanceUID": ["x", "y"]}))
    sub_rows = [{"StudyInstanceUID": "x", **{c: 0.5 for c in LABEL_COLS}}]
    sub_p = write_csv(tmp_path / "sub.csv", pd.DataFrame(sub_rows))
    assert len(load_test(test_p)) == 2
    sub = load_sample_submission(sub_p)
    assert list(sub.columns) == ["StudyInstanceUID", *LABEL_COLS]


def test_label_statistics_deterministic():
    a = pd.DataFrame([s.as_dict() for s in label_statistics(sample_train())])
    b = pd.DataFrame([s.as_dict() for s in label_statistics(sample_train())])
    assert a.equals(b)


def test_real_dataset_schema_holds():
    raw = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.csv"
    if not raw.exists():
        pytest.skip("real dataset not present")
    df = load_train(raw)
    assert df["StudyInstanceUID"].is_unique
