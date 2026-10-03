"""Aspect-preserving letterbox resize to the model input size.

ENGINEERING DECISION: centered padding with a configurable constant fill.
No cropping is applied (Stage-1 FACT: 640x540 non-square matrices exist, and
aggressive cropping could remove anatomy). Scale/pad metadata is returned so
provenance stays complete.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from orthovision.preprocessing.config import PreprocessConfig


class ResizeError(ValueError):
    pass


@dataclass(frozen=True)
class ResizeMeta:
    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    scale: float
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    interpolation: str
    fill: float


def _interpolation_mode(name: str) -> str:
    if name == "bilinear":
        return "bilinear"
    if name == "nearest":
        return "nearest"
    raise ResizeError(f"unsupported resize interpolation: {name}")


def letterbox_resize(
    image: np.ndarray,
    cfg: PreprocessConfig,
) -> tuple[np.ndarray, ResizeMeta]:
    """Resize a single 2-D float image preserving aspect ratio, centered pad."""
    if image.ndim != 2:
        raise ResizeError(f"expected 2-D image, got {image.shape}")
    out_h, out_w = cfg.image_size
    if out_h < 1 or out_w < 1:
        raise ResizeError(f"invalid target size: {cfg.image_size}")

    h, w = image.shape
    scale = min(out_h / h, out_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    mode = _interpolation_mode(cfg.resize_interpolation)
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=(new_h, new_w), mode=mode, align_corners=False if mode == "bilinear" else None)
    resized = resized[0, 0]

    pad_total_h = out_h - new_h
    pad_total_w = out_w - new_w
    pad_top = pad_total_h // 2
    pad_bottom = pad_total_h - pad_top
    pad_left = pad_total_w // 2
    pad_right = pad_total_w - pad_left

    if cfg.resize_fill != 0.0:
        base = torch.full((out_h, out_w), float(cfg.resize_fill), dtype=torch.float32)
        base[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
        out = base
    else:
        out = F.pad(
            resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )

    meta = ResizeMeta(
        original_shape=(h, w),
        resized_shape=(out_h, out_w),
        scale=float(scale),
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        interpolation=mode,
        fill=float(cfg.resize_fill),
    )
    return out.numpy().astype(np.float32), meta


def stack_triplet_channels(channels: list[np.ndarray], cfg: PreprocessConfig) -> tuple[np.ndarray, list[ResizeMeta]]:
    """Resize each channel independently and stack to [3, H, W]."""
    outs: list[np.ndarray] = []
    metas: list[ResizeMeta] = []
    for ch in channels:
        r, m = letterbox_resize(ch, cfg)
        outs.append(r)
        metas.append(m)
    return np.stack(outs, axis=0), metas
