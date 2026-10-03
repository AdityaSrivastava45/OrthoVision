"""Logging helpers for the preprocessing package."""

from __future__ import annotations

import logging


def log_cache_issue(key: str, exc: Exception) -> None:
    logging.getLogger("orthovision.preprocessing.cache").warning(
        "cache entry %s unreadable (%s); recomputing", key, exc
    )
