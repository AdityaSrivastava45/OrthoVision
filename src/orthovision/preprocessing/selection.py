"""Deterministic, configurable series ranking and selection.

A hard filter on anatomical plane is applied first; remaining candidates are
scored with transparent components in [0, 1] combined via configured weights.
Ties break deterministically: score desc -> slice count desc -> UID asc.

This ranking is an engineering baseline, not a medically validated optimum;
every weight lives in PreprocessConfig for later experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orthovision.preprocessing.config import PreprocessConfig
from orthovision.preprocessing.discovery import SeriesInfo


@dataclass
class CandidateScore:
    series_uid: str
    score: float
    reasons: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class SelectionReport:
    plane: str
    fluid_preference: int | None
    selected_series_uid: str | None
    candidates: list[CandidateScore]
    notes: list[str] = field(default_factory=list)

    @property
    def selected(self) -> CandidateScore | None:
        if not self.candidates or self.selected_series_uid is None:
            return None
        for c in self.candidates:
            if c.series_uid == self.selected_series_uid:
                return c
        return None


def _saturating(x: float) -> float:
    if x <= 0:
        return 0.0
    return float(x / (1.0 + x))


def _components(info: SeriesInfo, cfg: PreprocessConfig) -> dict[str, float]:
    slices_component = _saturating(max(0, info.n_slices - cfg.min_slices_usable) / 45.0)
    area = (info.rows or 0) * (info.columns or 0)
    resolution_component = _saturating(area / (512.0 * 512.0))
    completeness = max(0.0, min(1.0, info.completeness))

    geometry = info.geometry
    consistency = 0.5
    if geometry.ordering_method == "spatial" and info.plane_derived is not None:
        consistency = 1.0
        if geometry.gap_cv is not None and geometry.gap_cv > cfg.unusual_spacing_cv:
            consistency -= 0.5
        if geometry.duplicate_positions > 0:
            consistency -= 0.25
        if info.plane_agrees() is False:
            consistency -= 0.25
    elif info.plane_agrees() is False:
        consistency = 0.25

    return {
        "slices": round(slices_component, 6),
        "resolution": round(resolution_component, 6),
        "completeness": round(completeness, 6),
        "consistency": round(max(0.0, min(1.0, consistency)), 6),
    }


def rank_candidates(
    candidates: list[SeriesInfo],
    *,
    plane: str,
    fluid_preference: int | None,
    cfg: PreprocessConfig,
) -> SelectionReport:
    weights = cfg.ranking_weights

    hard_plane = [c for c in candidates if (c.plane_csv == plane or c.plane_derived == plane)]
    notes: list[str] = []
    if not hard_plane:
        return SelectionReport(
            plane=plane,
            fluid_preference=fluid_preference,
            selected_series_uid=None,
            candidates=[],
            notes=[f"no candidate series matches plane {plane}"],
        )

    scored: list[CandidateScore] = []
    for c in hard_plane:
        comps = _components(c, cfg)
        reasons: list[str] = []
        total = 0.0
        if fluid_preference is not None and c.fluid_sensitive is not None:
            match = 1.0 if int(c.fluid_sensitive) == int(fluid_preference) else 0.0
            w = float(weights.get("fluid", 0.0))
            total += w * match
            comps["fluid"] = match
            reasons.append(
                f"fluid_sensitive={int(c.fluid_sensitive)} {'==' if match else '!='} preference {fluid_preference}"
            )
        for key in ("slices", "resolution", "completeness", "consistency"):
            w = float(weights.get(key, 0.0))
            total += w * comps[key]
            reasons.append(f"{key}={comps[key]:.3f} x w={w:g}")
        scored.append(CandidateScore(series_uid=c.series_uid, score=round(total, 6), reasons=reasons, components=comps))

    slice_counts = {c.series_uid: c.n_slices for c in hard_plane}
    scored.sort(key=lambda s: (-s.score, -slice_counts[s.series_uid], s.series_uid))
    return SelectionReport(
        plane=plane,
        fluid_preference=fluid_preference,
        selected_series_uid=scored[0].series_uid,
        candidates=scored,
        notes=notes,
    )


def select_study_series(study_series: list[SeriesInfo], cfg: PreprocessConfig) -> dict[str, SelectionReport]:
    """One selection report per configured plane slot."""
    reports: dict[str, SelectionReport] = {}
    for plane in cfg.planes:
        pref = cfg.fluid_preference.get(plane)
        reports[plane] = rank_candidates(study_series, plane=plane, fluid_preference=pref, cfg=cfg)
    return reports
