"""Project-configurable paths and inspection thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until pyproject.toml is found."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here


@dataclass(frozen=True)
class DataConfig:
    dataset_root: Path
    output_dir: Path
    manifest_dir: Path
    viz_dir: Path
    num_workers: int
    logging_level: str
    log_file: Path | None
    plane_min_alignment: float
    plane_min_axis_separation: float
    min_slices_warning: int
    unusual_spacing_cv: float
    duplicate_position_epsilon_mm: float
    orientation_epsilon: float
    project_root: Path

    def resolve(self, path: Path | str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()


def _as_path(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def load_data_config(path: Path | str | None = None, *, project_root: Path | None = None) -> DataConfig:
    root = project_root or find_project_root()
    cfg_path = Path(path) if path else root / "configs" / "data.yaml"
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve() if not cfg_path.exists() else cfg_path.resolve()

    raw: dict[str, Any] = {}
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config at {cfg_path} must be a mapping")
            raw = loaded

    def rel(key: str, default: str) -> Path:
        value = Path(_as_path(raw.get(key), default))
        return value if value.is_absolute() else (root / value).resolve()

    log_raw = raw.get("log_file")
    log_file: Path | None
    if log_raw in (None, "", False):
        log_file = None
    else:
        lp = Path(str(log_raw))
        log_file = lp if lp.is_absolute() else (root / lp).resolve()

    return DataConfig(
        dataset_root=rel("dataset_root", "data/raw"),
        output_dir=rel("output_dir", "outputs"),
        manifest_dir=rel("manifest_dir", "outputs/manifests"),
        viz_dir=rel("viz_dir", "outputs/viz"),
        num_workers=int(raw.get("num_workers", 4)),
        logging_level=str(raw.get("logging_level", "INFO")).upper(),
        log_file=log_file,
        plane_min_alignment=float(raw.get("plane_min_alignment", 0.8)),
        plane_min_axis_separation=float(raw.get("plane_min_axis_separation", 0.2)),
        min_slices_warning=int(raw.get("min_slices_warning", 4)),
        unusual_spacing_cv=float(raw.get("unusual_spacing_cv", 0.15)),
        duplicate_position_epsilon_mm=float(raw.get("duplicate_position_epsilon_mm", 1e-3)),
        orientation_epsilon=float(raw.get("orientation_epsilon", 1e-3)),
        project_root=root,
    )
