"""Near-real-time (daily micro-batch) incremental scoring for UEBA.

Freezes the baselines and model once, then scores new days against them and
appends to ``risk.parquet`` - so the dashboard reflects new data as it lands,
at daily cadence, offline. See :mod:`src.ueba.live.incremental`.
"""

from .incremental import (
    BASELINES_FILENAME,
    baselines_path,
    freeze_baselines,
    incremental_update,
    load_baselines,
    save_baselines,
    score_against_baselines,
)

__all__ = [
    "BASELINES_FILENAME",
    "baselines_path",
    "freeze_baselines",
    "incremental_update",
    "load_baselines",
    "save_baselines",
    "score_against_baselines",
]
