"""Study-level data splitting with patient awareness and leakage prevention."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from orthovision.data.tabular import load_train

log = logging.getLogger("orthovision.training.split")


@dataclass
class SplitConfig:
    """Configuration for study split."""
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    seed: int = 42
    patient_aware: bool = True  # Group by PatientID if available
    output_dir: str = "splits"


@dataclass
class StudySplit:
    """Container for study split."""
    train_study_ids: list[str]
    val_study_ids: list[str]
    config: SplitConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_study_ids": self.train_study_ids,
            "val_study_ids": self.val_study_ids,
            "config": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "seed": self.config.seed,
                "patient_aware": self.config.patient_aware,
            }
        }

    def save(self, path: str | Path) -> None:
        """Save split to CSV files."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"StudyInstanceUID": self.train_study_ids}).to_csv(
            path / "train.csv", index=False
        )
        pd.DataFrame({"StudyInstanceUID": self.val_study_ids}).to_csv(
            path / "val.csv", index=False
        )
        log.info(f"Split saved to {path}: train={len(self.train_study_ids)}, val={len(self.val_study_ids)}")

    @classmethod
    def load(cls, path: str | Path, config: SplitConfig | None = None) -> "StudySplit":
        """Load split from CSV files."""
        path = Path(path)
        train_df = pd.read_csv(path / "train.csv")
        val_df = pd.read_csv(path / "val.csv")

        if config is None:
            config = SplitConfig()

        return cls(
            train_study_ids=train_df["StudyInstanceUID"].tolist(),
            val_study_ids=val_df["StudyInstanceUID"].tolist(),
            config=config,
        )


def get_patient_ids(study_ids: list[str], labels_csv: str | Path) -> dict[str, str]:
    """Get PatientID for each study from DICOM metadata or labels CSV.

    Tries to load from DICOM metadata first, then falls back to assuming
    one study per patient (study_id as patient_id).

    Args:
        study_ids: list of StudyInstanceUIDs
        labels_csv: path to train.csv (may have PatientID column)

    Returns:
        Dict mapping StudyInstanceUID -> PatientID
    """
    # Try to load from labels CSV
    try:
        df = pd.read_csv(labels_csv)
        if "PatientID" in df.columns:
            patient_map = df.set_index("StudyInstanceUID")["PatientID"].to_dict()
            return {sid: patient_map.get(sid, sid) for sid in study_ids}
    except Exception:
        pass

    # Fallback: each study is its own patient
    log.warning("PatientID not found in labels CSV, using StudyInstanceUID as patient identifier")
    return {sid: sid for sid in study_ids}


def create_study_split(
    study_ids: list[str],
    config: SplitConfig,
    labels_csv: str | Path | None = None,
) -> StudySplit:
    """Create train/validation split at study level.

    Args:
        study_ids: list of all StudyInstanceUIDs
        config: SplitConfig
        labels_csv: optional path to train.csv for patient-aware splitting

    Returns:
        StudySplit object
    """
    np.random.seed(config.seed)

    if config.patient_aware and labels_csv:
        # Group by patient
        patient_map = get_patient_ids(study_ids, labels_csv)
        patients = list(set(patient_map.values()))
        log.info(f"Found {len(patients)} unique patients for {len(study_ids)} studies")

        # Shuffle patients
        np.random.shuffle(patients)

        # Split patients
        n_train_patients = int(len(patients) * config.train_ratio)
        train_patients = set(patients[:n_train_patients])
        val_patients = set(patients[n_train_patients:])

        train_study_ids = [sid for sid in study_ids if patient_map[sid] in train_patients]
        val_study_ids = [sid for sid in study_ids if patient_map[sid] in val_patients]

        log.info(f"Patient-aware split: {len(train_patients)} train patients, "
                 f"{len(val_patients)} val patients")
    else:
        # Simple study-level split
        shuffled = study_ids.copy()
        np.random.shuffle(shuffled)
        n_train = int(len(shuffled) * config.train_ratio)
        train_study_ids = shuffled[:n_train]
        val_study_ids = shuffled[n_train:]
        log.info(f"Study-level split: {len(train_study_ids)} train, {len(val_study_ids)} val")

    # Verify no leakage
    verify_no_leakage(train_study_ids, val_study_ids)

    return StudySplit(
        train_study_ids=train_study_ids,
        val_study_ids=val_study_ids,
        config=config,
    )


def verify_no_leakage(train_ids: list[str], val_ids: list[str]) -> None:
    """Verify no study ID appears in both train and validation.

    Raises:
        ValueError: if leakage detected
    """
    train_set = set(train_ids)
    val_set = set(val_ids)
    intersection = train_set & val_set
    if intersection:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(intersection)} studies in both train and val: "
            f"{list(intersection)[:5]}..."
        )
    log.info("Leakage check passed: no overlapping study IDs")


def load_split(path: str | Path, config: SplitConfig | None = None) -> "StudySplit":
    """Load split from CSV files."""
    path = Path(path)
    train_df = pd.read_csv(path / "train.csv")
    val_df = pd.read_csv(path / "val.csv")

    if config is None:
        config = SplitConfig()

    return StudySplit(
        train_study_ids=train_df["StudyInstanceUID"].tolist(),
        val_study_ids=val_df["StudyInstanceUID"].tolist(),
        config=config,
    )


def get_training_labels(
    train_study_ids: list[str],
    labels_csv: str | Path,
) -> np.ndarray:
    """Get labels for training studies only.

    Args:
        train_study_ids: list of training StudyInstanceUIDs
        labels_csv: path to train.csv

    Returns:
        [N_train, 12] numpy array of labels (0/1/NaN)
    """
    df = load_train(labels_csv)
    df_train = df[df["StudyInstanceUID"].isin(train_study_ids)]

    # Get labels in correct order
    from orthovision.data.tabular import LABELS
    # Convert tuple to list for pandas column indexing (tuple triggers MultiIndex lookup)
    labels = df_train[list(LABELS)].to_numpy(dtype=np.float32)
    return labels