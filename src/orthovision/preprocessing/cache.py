"""Config-keyed cache for processed 2.5D samples.

Cache key = sha256(config fingerprint + study/series UIDs + ordered SOP UIDs).
Any change to preprocessing configuration automatically produces a new key,
so stale entries are never read. Storage is one .npz per series.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orthovision.preprocessing.config import PreprocessConfig

PIPELINE_VERSION = "2"

def config_fingerprint(cfg: PreprocessConfig) -> str:
    return cfg.fingerprint()


def series_cache_key(cfg: PreprocessConfig, series_uid: str, sop_uids: list[str]) -> str:
    payload = json.dumps(
        {
            "pipeline_version": PIPELINE_VERSION,
            "config": cfg.fingerprint(),
            "series_uid": series_uid,
            "sops": sop_uids,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class CachedSeries:
    images: np.ndarray  # [N, 3, H, W] float32
    metadata_json: str


class SampleCache:
    def __init__(self, directory: Path | str | None, enabled: bool = True):
        self.enabled = bool(enabled) and directory is not None
        self.directory = Path(directory) if directory else None
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        assert self.directory is not None
        return self.directory / f"{key}.npz"

    def get(self, key: str) -> CachedSeries | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                return CachedSeries(images=z["images"], metadata_json=str(z["metadata_json"]))
        except Exception as exc:  # noqa: BLE001 - corrupt cache must not crash runs
            from orthovision.preprocessing.logutil import log_cache_issue

            log_cache_issue(key, exc)
            return None

    def put(self, key: str, images: np.ndarray, metadata_json: str) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        fd, tmp = tempfile.mkstemp(suffix=".npz", dir=str(self.directory))
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, images=images.astype(np.float32), metadata_json=metadata_json)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def invalidate_series(self, cfg: PreprocessConfig, series_uid: str, sop_uids: list[str]) -> None:
        """Explicit invalidation helper (automatic via keys otherwise)."""
        key = series_cache_key(cfg, series_uid, sop_uids)
        if self.enabled and self._path(key).exists():
            self._path(key).unlink()
