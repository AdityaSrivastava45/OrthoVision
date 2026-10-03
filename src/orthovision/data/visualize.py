"""Research/debug visualization of ordered slices. Not a product UI."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orthovision.data.logutil import get_logger
from orthovision.data.models import Series, Study
from orthovision.data.pixels import PixelDecodeError, decode_path, display_preview

log = get_logger("visualize")


def _pick_indices(n: int, k: int) -> list[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, k)})


def series_mosaic(
    series: Series,
    *,
    n_slices: int = 5,
    apply_window: bool = False,
):
    """Return (figure, axes) showing evenly spaced ordered slices."""
    idxs = _pick_indices(series.slice_count, n_slices)
    fig, axes = plt.subplots(1, max(len(idxs), 1), figsize=(3.2 * max(len(idxs), 1), 3.4))
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    title = (
        f"{series.series_description or '(no SeriesDescription)'} | "
        f"plane={series.plane} | {series.slice_count} slices | "
        f"{series.validation.status if series.validation else '?'}"
    )
    fig.suptitle(title, fontsize=10)
    if not idxs:
        axes[0].set_title("empty series")
        axes[0].axis("off")
        return fig, axes
    for ax, i in zip(axes, idxs):
        sl = series.slices[i]
        ax.set_title(f"idx {i}/{series.slice_count - 1}\n{sl.sop_instance_uid[-10:]}", fontsize=8)
        try:
            payload = decode_path(sl.source_path)
            img = display_preview(payload, apply_window=apply_window)
            if img.ndim == 3:
                img = img[img.shape[0] // 2]
            ax.imshow(img, cmap="gray")
        except PixelDecodeError as exc:
            ax.text(0.5, 0.5, f"decode error\n{exc}", ha="center", va="center", wrap=True, fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def visualize_study(
    study: Study,
    dest_dir: Path,
    *,
    n_slices: int = 5,
    include_invalid: bool = False,
) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ser in study.series:
        status = ser.validation.status if ser.validation else "INVALID"
        if status == "INVALID" and not include_invalid:
            log.info("Skipping INVALID series %s for viz", ser.series_id)
            continue
        fig, _ = series_mosaic(ser, n_slices=n_slices)
        safe = ser.series_id.replace(".", "_")[-40:]
        path = dest_dir / f"{study.study_id[-12:]}_{safe}_{ser.plane}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
        log.info("Wrote %s", path)
    return written
