"""Physical-space sampling and 2.5D triplet construction.

Locations are chosen in the *ordered* series so they correspond to even
anatomical spread, not arbitrary file order. Boundary slices are avoided by a
configurable margin. Triplet neighbours follow a configurable strategy:
'index' (offsets) or 'physical_distance' (mm-based search, extensible).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orthovision.preprocessing.config import PreprocessConfig


class SamplingError(ValueError):
    pass


@dataclass(frozen=True)
class TripletPlan:
    center_index: int
    slice_indices: tuple[int, int, int]
    physical_positions_mm: tuple[float | None, float | None, float | None]
    channel_roles: tuple[str, str, str] = ("previous", "center", "next")


def sample_locations(
    n_slices: int,
    cfg: PreprocessConfig,
    *,
    positions_mm: list[float] | None = None,
) -> list[int]:
    """Deterministically choose center-slice indices across the stack."""
    if n_slices < 1:
        return []
    if cfg.sample_positions is not None:
        fracs = list(cfg.sample_positions)
        clamp = False
    else:
        margin = float(np.clip(cfg.boundary_margin, 0.0, 0.49))
        fracs = np.linspace(margin, 1.0 - margin, cfg.num_samples_per_plane).tolist()
        clamp = True

    indices: list[int] = []
    seen: set[int] = set()
    lo_bound = 1 if clamp and n_slices >= 3 else 0
    hi_bound = n_slices - 2 if clamp and n_slices >= 3 else n_slices - 1
    for f in fracs:
        if not 0.0 <= f <= 1.0:
            raise SamplingError(f"sample position fraction {f} outside [0, 1]")
        idx = int(round(float(f) * (n_slices - 1)))
        idx = int(np.clip(idx, lo_bound, hi_bound))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def _nearest_index(positions: np.ndarray, target: float, center: int, max_search: int) -> int | None:
    lo = max(0, center - max_search)
    hi = min(len(positions), center + max_search + 1)
    window = positions[lo:hi]
    j = int(np.argmin(np.abs(window - target))) + lo
    if j == center:
        return None
    return j


def build_triplet(
    center_index: int,
    n_slices: int,
    cfg: PreprocessConfig,
    positions_mm: list[float] | None,
) -> TripletPlan:
    offsets = tuple(cfg.neighbor_offsets)
    if len(offsets) != 3 or offsets[1] != 0 or not (offsets[0] < 0 < offsets[2]):
        raise SamplingError(
            "neighbor_offsets must be (negative, 0, positive), e.g. (-1, 0, 1)"
        )

    prev_i = next_i = None
    if cfg.neighbor_strategy == "index":
        p, c, nx = center_index + offsets[0], center_index, center_index + offsets[2]
        if 0 <= p < n_slices and 0 <= nx < n_slices:
            prev_i, next_i = p, nx
    elif cfg.neighbor_strategy == "physical_distance":
        if positions_mm is None or len(positions_mm) != n_slices:
            raise SamplingError("physical_distance strategy requires positions for every slice")
        pos = np.asarray(positions_mm, dtype=np.float64)
        finite = [v for v in positions_mm if v is not None]
        median_gap = float(np.median(np.abs(np.diff(np.sort(finite))))) if len(finite) >= 2 else 1.0
        max_search = cfg.neighbor_max_search_slices
        step = cfg.neighbor_target_spacing_mm if cfg.neighbor_target_spacing_mm else median_gap
        p_target = pos[center_index] + offsets[0] * step
        n_target = pos[center_index] + offsets[2] * step
        prev_i = _nearest_index(pos, min(p_target, n_target), center_index, max_search)
        next_i = _nearest_index(pos, max(p_target, n_target), center_index, max_search)
    else:
        raise SamplingError(f"unknown neighbor strategy: {cfg.neighbor_strategy}")

    if prev_i is None or next_i is None or len({prev_i, center_index, next_i}) != 3:
        raise SamplingError(
            f"cannot form full triplet at center {center_index} "
            f"(n={n_slices}, strategy={cfg.neighbor_strategy})"
        )

    positions_out: tuple[float | None, float | None, float | None] = (None, None, None)
    if positions_mm is not None and len(positions_mm) == n_slices:
        positions_out = (positions_mm[prev_i], positions_mm[center_index], positions_mm[next_i])

    return TripletPlan(
        center_index=center_index,
        slice_indices=(prev_i, center_index, next_i),
        physical_positions_mm=positions_out,
    )


def plan_series_samples(
    n_slices: int,
    positions_mm: list[float] | None,
    cfg: PreprocessConfig,
) -> list[TripletPlan]:
    """All triplets for one series; locations that cannot form a full
    triplet are reported by callers via SamplingError handling."""
    plans: list[TripletPlan] = []
    for center in sample_locations(n_slices, cfg, positions_mm=positions_mm):
        try:
            plans.append(build_triplet(center, n_slices, cfg, positions_mm))
        except SamplingError:
            continue
    return plans
