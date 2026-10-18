"""Ingestion layer: chunked read of raw CSVs and daily aggregation.

Exposes :func:`ingest`, which reads every raw event CSV a chunk at a time and
boils them down to one row per ``(user, calendar_day)`` in
``user_day.parquet`` (see :mod:`src.ueba.ingestion.pipeline`), plus the subset
helpers :func:`select_subset` and :func:`ingest_subset` that restrict that
output to the working Subset — all insiders plus a reproducible sample of
normal users (see :mod:`src.ueba.ingestion.subset`).
"""

from .pipeline import (
    USER_DAY_FILENAME,
    ingest,
)
from .subset import (
    ingest_subset,
    read_insiders,
    read_population,
    select_subset,
)

__all__ = [
    "ingest",
    "USER_DAY_FILENAME",
    "select_subset",
    "ingest_subset",
    "read_insiders",
    "read_population",
]
