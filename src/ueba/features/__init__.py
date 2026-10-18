"""Features layer: per-user-day behavioral feature extraction.

Exposes :func:`extract_features`, which reads the raw event CSVs a chunk at a
time and computes the full behavioural feature table described in the design's
Data Models section — the logon, USB/removable-media, web, and email families —
one row per ``(user, calendar_day)``, written to ``user_day.parquet`` (see
:mod:`src.ueba.features.extractor`). After-hours, weekend, and site-category
classifications all come from ``config``; nothing is hard-coded.
"""

from .extractor import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    USER_DAY_FILENAME,
    extract_features,
)

__all__ = [
    "extract_features",
    "FEATURE_COLUMNS",
    "OUTPUT_COLUMNS",
    "USER_DAY_FILENAME",
]
