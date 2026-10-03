"""Preprocessing configuration for the MRI -> 2.5D pipeline.

Every knob that Stage 3+ experiments may want to vary lives here and in
configs/preprocessing.yaml. Nothing is hardcoded at call sites.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

NormalizationMethod = Literal["percentile", "zscore"]
NeighborStrategy = Literal["index", "physical_distance"]
RunMode = Literal["strict", "permissive"]

DEFAULT_CONFIG_PATH = Path("configs") / "preprocessing.yaml"

RANKING_COMPONENT_KEYS = (
    "slices",
    "resolution",
    "completeness",
    "consistency",
)


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: tuple[int, int] = (224, 224)
    resize_interpolation: str = "bilinear"
    resize_fill: float = 0.0

    normalization_method: NormalizationMethod = "percentile"
    normalization_scope: Literal["slice", "triplet"] = "slice"
    clip_percentiles: tuple[float, float] = (0.5, 99.5)

    planes: tuple[str, ...] = ("Sagittal", "Coronal", "Axial")
    fluid_preference: dict[str, int | None] = field(
        default_factory=lambda: {
            "Sagittal": 1,
            "Coronal": 1,
            "Axial": None,
        }
    )

    num_samples_per_plane: int = 6
    sample_positions: tuple[float, ...] | None = None
    boundary_margin: float = 0.06

    neighbor_strategy: NeighborStrategy = "index"
    neighbor_offsets: tuple[int, int, int] = (-1, 0, 1)
    neighbor_target_spacing_mm: float | None = None
    neighbor_max_search_slices: int = 5

    ranking_weights: dict[str, float] = field(
        default_factory=lambda: {
            "fluid": 1.5,
            "slices": 1.0,
            "resolution": 0.5,
            "completeness": 1.0,
            "consistency": 2.0,
        }
    )

    min_slices_usable: int = 3
    duplicate_position_epsilon_mm: float = 1.0e-3
    unusual_spacing_cv: float = 0.15
    orientation_min_alignment: float = 0.8

    mode: RunMode = "permissive"
    cache_enabled: bool = True
    cache_dir: Path | None = None

    def with_overrides(self, **overrides: Any) -> "PreprocessConfig":
        return dataclasses.replace(self, **overrides)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, Path):
                value = str(value)
            elif isinstance(value, dict):
                value = {k: (list(v) if isinstance(v, tuple) else v) for k, v in value.items()}
            out[f.name] = value
        return out


def _get(cfg_dict: dict[str, Any], dotted: str, default: Any) -> Any:
    node: Any = cfg_dict
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def load_preprocess_config(path: Path | str | None = None) -> PreprocessConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return PreprocessConfig()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    size = _get(raw, "image_size", [224, 224])
    positions = _get(raw, "sampling.sample_positions", None)

    fluid_pref_raw = _get(raw, "selection.fluid_preference", {})
    fluid_pref = {"Sagittal": None, "Coronal": None, "Axial": None}
    for k, v in fluid_pref_raw.items():
        fluid_pref[k] = None if v in ("none", None) else int(v)

    weights_raw = _get(raw, "selection.ranking_weights", {})
    weights = {k: float(v) for k, v in weights_raw.items()}

    cache_dir_raw = _get(raw, "cache.dir", None)

    return PreprocessConfig(
        image_size=(int(size[0]), int(size[1])),
        resize_interpolation=str(_get(raw, "resize.interpolation", "bilinear")),
        resize_fill=float(_get(raw, "resize.fill", 0.0)),
        normalization_method=str(_get(raw, "normalization.method", "percentile")),
        normalization_scope=str(_get(raw, "normalization.scope", "slice")),
        clip_percentiles=tuple(float(v) for v in _get(raw, "normalization.clip_percentiles", [0.5, 99.5])),
        planes=tuple(_get(raw, "planes", ["Sagittal", "Coronal", "Axial"])),
        fluid_preference=fluid_pref,
        num_samples_per_plane=int(_get(raw, "sampling.num_samples_per_plane", 6)),
        sample_positions=tuple(float(v) for v in positions) if positions else None,
        boundary_margin=float(_get(raw, "sampling.boundary_margin", 0.06)),
        neighbor_strategy=str(_get(raw, "neighbors.strategy", "index")),
        neighbor_offsets=tuple(int(v) for v in _get(raw, "neighbors.offsets", [-1, 0, 1])),
        neighbor_target_spacing_mm=_get(raw, "neighbors.target_spacing_mm", None),
        neighbor_max_search_slices=int(_get(raw, "neighbors.max_search_slices", 5)),
        ranking_weights=weights,
        min_slices_usable=int(_get(raw, "qc.min_slices_usable", 3)),
        duplicate_position_epsilon_mm=float(_get(raw, "qc.duplicate_position_epsilon_mm", 1e-3)),
        unusual_spacing_cv=float(_get(raw, "qc.unusual_spacing_cv", 0.15)),
        orientation_min_alignment=float(_get(raw, "qc.orientation_min_alignment", 0.8)),
        mode=str(_get(raw, "mode", "permissive")),
        cache_enabled=bool(_get(raw, "cache.enabled", True)),
        cache_dir=Path(cache_dir_raw) if cache_dir_raw else None,
    )
