"""PyTorch-compatible study-level dataset over the Stage-2 pipeline.

One __getitem__ returns ALL 2.5D samples for one study:

    {
      "images":          FloatTensor [N, 3, H, W],
      "sample_metadata": list of provenance dicts (len N),
      "labels":          FloatTensor [12]  (NaN where unlabeled),
      "study_uid":       str,
    }

Modes:
    train     - transform hook may run (disabled by default -> deterministic)
    val       - deterministic, labels used if present
    inference - identical outputs; labels are NaN when absent

Unlabeled studies keep their place in the index; missing labels are NaN,
never imputed and never silently dropped (Stage-1 FACT: only 58/4407 studies
carry structured labels).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from orthovision.data.tabular import LABELS, load_train
from orthovision.preprocessing.config import PreprocessConfig, load_preprocess_config
from orthovision.preprocessing.discovery import SeriesInfo, discover_studies
from orthovision.preprocessing.pipeline import StudyPreprocessor

log = logging.getLogger("orthovision.preprocessing.dataset")

DatasetMode = str  # "train" | "val" | "inference"


class KneeStudyDataset(Dataset):
    def __init__(
        self,
        study_ids: list[str],
        *,
        data_root: Path | str,
        series_csv: Path | str,
        headers_cache: Path | str,
        labels_csv: Path | str | None = None,
        cfg: PreprocessConfig | None = None,
        mode: DatasetMode = "train",
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        super().__init__()
        self.cfg = cfg or load_preprocess_config()
        self.mode = mode
        self.transform = transform
        self.data_root = Path(data_root)
        self.study_ids = sorted(study_ids)

        self.studies: dict[str, list[SeriesInfo]] = discover_studies(headers_cache, series_csv, self.cfg)
        missing = [s for s in self.study_ids if s not in self.studies]
        if missing:
            raise KeyError(f"studies without discovered series on disk: {missing[:5]} ({len(missing)} total)")

        self.labels: pd.DataFrame | None = None
        if labels_csv is not None and Path(labels_csv).exists():
            self.labels = load_train(labels_csv)
            label_map = self.labels.set_index("StudyInstanceUID")
            self._label_frame = label_map
        else:
            self._label_frame = None

        self.preprocessor = StudyPreprocessor(self.cfg, data_root=self.data_root)
        self._results: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.study_ids)

    def _labels_for(self, study_uid: str) -> torch.Tensor:
        values = np.full(len(LABELS), np.nan, dtype=np.float32)
        if self._label_frame is not None and study_uid in self._label_frame.index:
            row = self._label_frame.loc[study_uid]
            for i, name in enumerate(LABELS):
                v = row.get(name)
                if v is not None and not pd.isna(v):
                    values[i] = float(v)
        return torch.from_numpy(values)

    def __getitem__(self, index: int) -> dict:
        study_uid = self.study_ids[index]
        result = self.preprocessor.process_study(study_uid, self.studies[study_uid])

        images: list[np.ndarray] = []
        metas: list[dict] = []
        for sample in result.samples:
            img = sample.image
            if self.transform is not None and self.mode == "train":
                img = self.transform(img)
            images.append(img)
            metas.append(sample.metadata)

        if images:
            tensor = torch.from_numpy(np.stack(images, axis=0)).float()
        else:
            tensor = torch.empty((0, 3, *self.cfg.image_size), dtype=torch.float32)

        return {
            "images": tensor,
            "sample_metadata": metas,
            "labels": self._labels_for(study_uid),
            "study_uid": study_uid,
        }
