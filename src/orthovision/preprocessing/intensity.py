"""MRI intensity processing for model inputs.

Pipeline per slice: rescaled array (slope/intercept already applied by
orthovision.data.pixels) -> optional MONOCHROME1 inversion -> robust clipping
-> normalization. Raw DICOM files are never touched.

Default: percentile normalization to [0, 1] with 0.5/99.5 clipping.
ENGINEERING DECISION: bounded [0, 1] output suits patch-embedding backbones
and keeps letterbox padding value meaningful; z-score is supported for
experiments but is unbounded and its fill value must then be chosen carefully.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orthovision.preprocessing.config import PreprocessConfig


class NormalizationError(ValueError):
    pass


@dataclass(frozen=True)
class NormalizationStats:
    method: str
    scope: str
    clip_low: float
    clip_high: float
    mean: float | None = None
    std: float | None = None
    inverted_monochrome1: bool = False


def _invert_monochrome1(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    hi = float(np.max(finite))
    lo = float(np.min(finite))
    return hi - (arr - lo)


def _apply_transform(
    clipped: np.ndarray,
    *,
    method: str,
    lo: float,
    hi: float,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[np.ndarray, float | None, float | None]:
    if method == "percentile":
        denom = hi - lo
        out = (clipped - lo) / denom if denom > 0 else np.zeros_like(clipped)
        return out, None, None
    if method == "zscore":
        if std is not None and std > 0:
            out = (clipped - mean) / std
            return out, mean, std
        return np.zeros_like(clipped), mean, std
    raise NormalizationError(f"unknown normalization method: {method}")


def prepare_slice(
    rescaled: np.ndarray,
    *,
    photometric: str | None,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, NormalizationStats]:
    """Normalize one slice using its own statistics (scope='slice')."""
    if rescaled.ndim != 2:
        raise NormalizationError(f"expected 2-D slice, got shape {rescaled.shape}")
    arr = rescaled.astype(np.float64, copy=True)

    inverted = False
    if (photometric or "").strip().upper() == "MONOCHROME1":
        arr = _invert_monochrome1(arr)
        inverted = True

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32), NormalizationStats(
            method=cfg.normalization_method, scope="slice", clip_low=0.0, clip_high=0.0,
            inverted_monochrome1=inverted,
        )

    lo_p, hi_p = cfg.clip_percentiles
    lo, hi = np.percentile(finite, [lo_p, hi_p])
    clipped = np.clip(arr, lo, hi)

    mean = std = None
    if cfg.normalization_method == "zscore":
        mean, std = float(np.mean(clipped)), float(np.std(clipped))
    out, mean, std = _apply_transform(
        clipped, method=cfg.normalization_method, lo=float(lo), hi=float(hi),
        mean=mean, std=std,
    )
    stats = NormalizationStats(
        cfg.normalization_method, "slice", float(lo), float(hi), mean=mean, std=std,
        inverted_monochrome1=inverted,
    )
    return out.astype(np.float32), stats


def prepare_triplet(
    rescaled_slices: list[np.ndarray],
    *,
    photometrics: list[str | None],
    cfg: PreprocessConfig,
) -> tuple[list[np.ndarray], list[NormalizationStats]]:
    """Normalize a triplet either per-slice or with shared triplet statistics.

    scope='triplet' keeps relative intensity between previous/center/next
    channels; scope='slice' normalizes each channel independently.
    """
    assert len(rescaled_slices) == len(photometrics) == 3
    arrays: list[np.ndarray] = []
    inverted_flags: list[bool] = []
    for arr, photo in zip(rescaled_slices, photometrics):
        a = arr.astype(np.float64, copy=True)
        inverted = False
        if (photo or "").strip().upper() == "MONOCHROME1":
            a = _invert_monochrome1(a)
            inverted = True
        arrays.append(a)
        inverted_flags.append(inverted)

    if cfg.normalization_scope == "slice":
        outs_stats = [prepare_slice(a, photometric=None, cfg=cfg) for a in arrays]
        outs = [o for o, _ in outs_stats]
        stats = [
            NormalizationStats(s.method, s.scope, s.clip_low, s.clip_high, s.mean, s.std,
                               inverted_monochrome1=s.inverted_monochrome1 or inv)
            for (o, s), inv in zip(outs_stats, inverted_flags)
        ]
        return outs, stats

    finite = np.concatenate([a[np.isfinite(a)] for a in arrays]) if any(a.size for a in arrays) else np.array([])
    if finite.size == 0:
        zeros = [np.zeros_like(a, dtype=np.float32) for a in arrays]
        stats = [NormalizationStats(cfg.normalization_method, "triplet", 0.0, 0.0) for _ in arrays]
        return zeros, stats

    lo_p, hi_p = cfg.clip_percentiles
    lo, hi = np.percentile(finite, [lo_p, hi_p])
    mean = std = None
    if cfg.normalization_method == "zscore":
        clipped_all = np.clip(finite, lo, hi)
        mean, std = float(np.mean(clipped_all)), float(np.std(clipped_all))

    outs: list[np.ndarray] = []
    stats: list[NormalizationStats] = []
    for a, inv in zip(arrays, inverted_flags):
        clipped = np.clip(a, lo, hi)
        out, m, s = _apply_transform(clipped, method=cfg.normalization_method,
                                     lo=float(lo), hi=float(hi), mean=mean, std=std)
        outs.append(out.astype(np.float32))
        stats.append(NormalizationStats(cfg.normalization_method, "triplet", float(lo), float(hi),
                                        mean=m, std=s, inverted_monochrome1=inv))
    return outs, stats
