"""Safe pixel decoding with explicit raw vs rescaled vs display-preview arrays.

This layer does **not** choose the training normalization. Callers must pick
which array to use.

Transformations
    raw
        `dataset.pixel_array` as stored (after transfer-syntax decode).
    rescaled
        `raw * RescaleSlope + RescaleIntercept` when those tags exist;
        otherwise identical to raw promoted to float64.
    display_preview
        Min-max to [0, 1] for debug visualization only. MONOCHROME1 is inverted
        after rescale so that "hot" display matches typical viewer convention.
        **Not for model training.**

Windowing / VOI LUT
    WindowCenter/WindowWidth are read and returned as metadata. They are not
    applied unless `apply_window=True` on `display_preview`. Arbitrary MRI
    windows are not invented.

Transfer syntax
    Uncompressed little/big endian and Deflated Explicit VR are handled by
    pydicom. JPEG / JPEG-LS / JPEG2000 / RLE need optional extras:
    `pip install orthovision[jpeg]` (pylibjpeg family). Failures are errors,
    never silent skips.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.uid import UID

from orthovision.data.logutil import get_logger
from orthovision.data.tags import get_float, get_floats, get_str, transfer_syntax_uid

log = get_logger("pixels")

# Common uncompressed / encapsulated UIDs we expect pydicom (or extras) to handle.
_UNCOMPRESSED = {
    "1.2.840.10008.1.2",  # Implicit VR Little Endian
    "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
    "1.2.840.10008.1.2.2",  # Explicit VR Big Endian
    "1.2.840.10008.1.2.1.99",  # Deflated Explicit VR Little Endian
}

_JPEG_FAMILY_PREFIXES = (
    "1.2.840.10008.1.2.4.",  # JPEG / JPEG-LS / JPEG2000
    "1.2.840.10008.1.2.5",  # RLE
)


class PixelDecodeError(RuntimeError):
    """Pixel data could not be decoded. Surface this in validation reports."""


@dataclass(frozen=True)
class PixelPayload:
    raw: np.ndarray
    rescaled: np.ndarray
    rescale_slope: float | None
    rescale_intercept: float | None
    photometric: str | None
    window_center: float | None
    window_width: float | None
    transfer_syntax_uid: str | None
    applied: tuple[str, ...]


def classify_transfer_syntax(uid: str | None) -> str:
    if uid is None:
        return "unknown"
    if uid in _UNCOMPRESSED:
        return "uncompressed"
    if uid.startswith("1.2.840.10008.1.2.4") or uid.startswith("1.2.840.10008.1.2.5"):
        return "compressed"
    return "other"


def _window_value(ds: Dataset, keyword: str) -> float | None:
    values = get_floats(ds, keyword)
    if values:
        return float(values[0])
    return get_float(ds, keyword)


def decode_dataset(ds: Dataset) -> PixelPayload:
    ts = transfer_syntax_uid(ds)
    try:
        raw = np.asarray(ds.pixel_array)
    except Exception as exc:  # noqa: BLE001
        kind = classify_transfer_syntax(ts)
        hint = ""
        if kind == "compressed":
            hint = " Install optional extras: pip install 'orthovision[jpeg]' (pylibjpeg)."
        name = UID(ts).name if ts else "unknown"
        raise PixelDecodeError(
            f"Failed to decode PixelData (TransferSyntaxUID={ts} {name}).{hint} Underlying: {exc}"
        ) from exc

    slope = get_float(ds, "RescaleSlope")
    intercept = get_float(ds, "RescaleIntercept")
    applied: list[str] = ["pixel_array"]
    if slope is not None or intercept is not None:
        s = 1.0 if slope is None else slope
        b = 0.0 if intercept is None else intercept
        rescaled = raw.astype(np.float64) * s + b
        applied.append("rescale_slope_intercept")
    else:
        rescaled = raw.astype(np.float64)

    return PixelPayload(
        raw=raw,
        rescaled=rescaled,
        rescale_slope=slope,
        rescale_intercept=intercept,
        photometric=get_str(ds, "PhotometricInterpretation"),
        window_center=_window_value(ds, "WindowCenter"),
        window_width=_window_value(ds, "WindowWidth"),
        transfer_syntax_uid=ts,
        applied=tuple(applied),
    )


def decode_path(path: Path, *, force: bool = False) -> PixelPayload:
    try:
        ds = dcmread(path, force=force)
    except Exception as exc:  # noqa: BLE001
        raise PixelDecodeError(f"Could not read DICOM for pixels: {path}: {exc}") from exc
    return decode_dataset(ds)


def display_preview(
    payload: PixelPayload,
    *,
    apply_window: bool = False,
) -> np.ndarray:
    """Return float array in ~[0, 1] for matplotlib only."""
    img = payload.rescaled.astype(np.float64)
    notes = []
    if apply_window and payload.window_center is not None and payload.window_width is not None and payload.window_width > 0:
        lo = payload.window_center - payload.window_width / 2.0
        hi = payload.window_center + payload.window_width / 2.0
        img = np.clip(img, lo, hi)
        notes.append("voi_window")
    photo = (payload.photometric or "").upper()
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img, dtype=np.float64)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if photo == "MONOCHROME1":
        img = hi - (img - lo)
        lo, hi = float(np.min(img)), float(np.max(img))
        notes.append("monochrome1_invert")
    if hi <= lo:
        out = np.zeros_like(img, dtype=np.float64)
    else:
        out = (img - lo) / (hi - lo)
    return out
