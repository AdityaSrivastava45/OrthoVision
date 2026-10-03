"""Safe DICOM tag access. Missing tags are None, never invented."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from pydicom.dataset import Dataset

T = TypeVar("T")


def _has(ds: Dataset, keyword: str) -> bool:
    return keyword in ds and ds.get(keyword) is not None


def get_str(ds: Dataset, keyword: str) -> str | None:
    if not _has(ds, keyword):
        return None
    value = ds.get(keyword)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def get_int(ds: Dataset, keyword: str) -> int | None:
    if not _has(ds, keyword):
        return None
    try:
        return int(ds.get(keyword))
    except (TypeError, ValueError):
        return None


def get_float(ds: Dataset, keyword: str) -> float | None:
    if not _has(ds, keyword):
        return None
    try:
        return float(ds.get(keyword))
    except (TypeError, ValueError):
        return None


def get_floats(ds: Dataset, keyword: str, expected: int | None = None) -> tuple[float, ...] | None:
    if not _has(ds, keyword):
        return None
    raw = ds.get(keyword)
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        try:
            return (float(raw),)
        except (TypeError, ValueError):
            return None
    try:
        values = tuple(float(x) for x in raw)
    except (TypeError, ValueError):
        return None
    if expected is not None and len(values) != expected:
        return None
    return values


def get_uid(ds: Dataset, keyword: str) -> str | None:
    return get_str(ds, keyword)


def transfer_syntax_uid(ds: Dataset) -> str | None:
    meta = getattr(ds, "file_meta", None)
    if meta is None:
        return None
    return get_str(meta, "TransferSyntaxUID")


def optional_repr(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, tuple):
        return "[" + ", ".join(optional_repr(v) or "" for v in value) + "]"
    return str(value)
