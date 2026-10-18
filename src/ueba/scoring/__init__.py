"""Scoring layer: statistical z-scores, fusion, and reason records.

Exposes two components:

* the **Statistical_Scorer** — :func:`self_score` and :func:`peer_score` turn
  one user-day into an anomaly score in ``0``–``1`` against the user's own
  baseline and their role group's baseline, and :func:`score_statistical` /
  :func:`write_scores_stat` score a whole feature table and persist the result
  (with top-3 reason features) to ``scores_stat.parquet`` (see
  :mod:`src.ueba.scoring.statistical`); and
* the **Scoring_Engine** — :func:`scale_isolation`, :func:`daily_risk`,
  :func:`weekly_risk`, and :func:`reason_record` fuse the self, isolation, and
  peer signals into a ``0``–``100`` daily risk, a trailing-7 weekly risk, and a
  top-3 reason record, while :func:`compute_risk` / :func:`write_risk` run the
  fusion end-to-end and persist it to ``risk.parquet`` (see
  :mod:`src.ueba.scoring.engine`).
"""

from .statistical import (
    SCORES_STAT_FILENAME,
    Z_CAP,
    peer_score,
    score_statistical,
    self_score,
    write_scores_stat,
)
from .engine import (
    RISK_FILENAME,
    compute_risk,
    daily_risk,
    reason_record,
    scale_isolation,
    weekly_risk,
    write_risk,
)

__all__ = [
    "SCORES_STAT_FILENAME",
    "Z_CAP",
    "peer_score",
    "score_statistical",
    "self_score",
    "write_scores_stat",
    "RISK_FILENAME",
    "compute_risk",
    "daily_risk",
    "reason_record",
    "scale_isolation",
    "weekly_risk",
    "write_risk",
]
