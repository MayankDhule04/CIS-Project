"""Scoring_Engine — fuse the three signals into one explainable daily risk.

By the time control reaches this module the hard analytical work is already
done. The Statistical_Scorer (task 3.5) has turned every user-day into a
``self_score`` and a ``peer_score`` in ``0``–``1`` (with the top-3 contributing
features recorded), and the Isolation_Model (task 4.1) has produced a *raw*
anomaly score per user-day where higher means more anomalous. This module is
the place those three opinions become a single, analyst-facing number.

It does four things, each matching one bullet of the design's Scoring_Engine:

1. **Scale the isolation score** into ``0``–``1`` so it sits on the same footing
   as the two statistical scores (design Property 14, Req 6.3).
2. **Fuse** the three ``0``–``1`` scores into a ``Daily_Risk`` in ``0``–``100``
   using the configured weights, ``100 * (w_self*self + w_iso*iso +
   w_peer*peer)`` (design Property 15, Req 6.4/6.5).
3. **Roll** each user's daily risks into a ``Weekly_Risk`` — the trailing-7 sum
   of that user's daily risks (design Property 16, Req 6.6).
4. **Explain** each day with a Reason_Record: the top-3 features that pushed the
   day's self score up, each with its value, its baseline mean, and its z-score
   (design Property 17, Req 6.7).

The fused result — scores, daily risk, weekly risk, and reasons — is written to
``<processed_dir>/risk.parquet`` (Req 6.8), the single artifact the dashboard
and the evaluation harness both read.

Isolation-score scaling (design "Scale to iso_score 0-1")
---------------------------------------------------------
The IsolationForest emits an unbounded raw anomaly score; the design only
requires the *Isolation_Score* to lie in ``0``–``1``. We use a deterministic
**min-max** transform over the set of user-days being scored::

    iso_score = (raw - raw.min()) / (raw.max() - raw.min())

This maps the least anomalous day in the batch to ``0`` and the most anomalous
to ``1`` while preserving the forest's ordering, and it needs no trained
parameters. The result is clamped into ``[0, 1]`` as a final guard. When every
raw score is identical (``max == min``) the transform is undefined, so we define
the whole batch as ``0.5`` — a neutral midpoint, since with no spread there is
no relative anomaly signal to separate the days. Scaling is therefore *relative
to the scored batch*; this is the documented, intended behaviour.

Weekly-risk convention (design "sum of the last 7 daily_risk")
--------------------------------------------------------------
``Weekly_Risk`` for a user-day is the sum of that user's :func:`daily_risk`
values over the **trailing seven rows ordered by day** (the current day plus the
six prior scored days, fewer near the start of the series). This is a
trailing-7-*rows* window, matching Property 16's "sum of the last seven
Daily_Risk values"; it is not a trailing-7-calendar-days window, so gaps in a
user's active days do not dilute the sum.

What this module does **not** do: it does not train or persist the
IsolationForest (task 4.1), build baselines (task 3.3), compute the statistical
scores (task 3.5), evaluate detection quality (phase 5), or render the dashboard
(phase 6). It only imports those pieces and fuses their outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import Config, FusionWeights
from ..baselines import build_baselines
from ..models import IsolationModel, score as iso_score_model, train_from_features
from .statistical import TOP_K, score_statistical, self_score

__all__ = [
    "RISK_FILENAME",
    "scale_isolation",
    "daily_risk",
    "weekly_risk",
    "reason_record",
    "compute_risk",
    "write_risk",
]

# The artifact this engine writes (fused daily/weekly risk + reasons).
RISK_FILENAME = "risk.parquet"

# Length of the trailing window, in rows, used for Weekly_Risk.
WEEKLY_WINDOW = 7

# The score assigned to every row when the raw isolation scores have no spread
# (max == min), so min-max scaling is undefined. A neutral midpoint.
DEGENERATE_ISO_SCORE = 0.5

# Explicit Arrow types so the list-of-struct ``reasons`` column round-trips
# exactly, even for rows whose reason list is empty (which would otherwise give
# pyarrow nothing to infer the struct shape from). Mirrors the Statistical_
# Scorer's top-features struct so the two artifacts stay schema-compatible.
_REASON_STRUCT = pa.struct(
    [
        ("feature", pa.string()),
        ("value", pa.float64()),
        ("baseline_mean", pa.float64()),
        ("z_score", pa.float64()),
    ]
)

_RISK_SCHEMA = pa.schema(
    [
        ("user", pa.string()),
        ("day", pa.timestamp("ns")),
        ("self_score", pa.float64()),
        ("isolation_score", pa.float64()),
        ("peer_score", pa.float64()),
        ("daily_risk", pa.float64()),
        ("weekly_risk", pa.float64()),
        ("reasons", pa.list_(_REASON_STRUCT)),
    ]
)

# Column order of the risk frame, matching design.md's risk.parquet Data Model.
RISK_COLUMNS = [
    "user",
    "day",
    "self_score",
    "isolation_score",
    "peer_score",
    "daily_risk",
    "weekly_risk",
    "reasons",
]


def scale_isolation(raw_scores: Any) -> np.ndarray:
    """Scale raw IsolationForest anomaly scores into the range ``0``–``1``.

    Applies a deterministic min-max transform over ``raw_scores`` (an array,
    list, or Series of per-user-day raw scores where higher = more anomalous):
    the minimum raw score maps to ``0`` and the maximum to ``1``, preserving the
    forest's ordering. The result is clamped into ``[0, 1]`` as a final guard so
    the Isolation_Score is always bounded (design Property 14, Req 6.3).

    When the batch has no spread (every raw score equal, so ``max == min``) the
    transform is undefined and the whole batch is defined as
    :data:`DEGENERATE_ISO_SCORE` (``0.5``) — a neutral midpoint, since with no
    spread no day is relatively more anomalous than another. An empty input
    returns an empty array.

    The scaling is relative to the scored batch, which is the intended,
    documented behaviour for turning an unbounded anomaly score into a bounded,
    rank-preserving Isolation_Score.
    """
    raw = np.asarray(raw_scores, dtype="float64").ravel()
    if raw.size == 0:
        return np.empty(0, dtype="float64")

    lo = float(np.nanmin(raw))
    hi = float(np.nanmax(raw))
    spread = hi - lo
    if not np.isfinite(spread) or spread == 0.0:
        return np.full(raw.shape, DEGENERATE_ISO_SCORE, dtype="float64")

    scaled = (raw - lo) / spread
    return np.clip(scaled, 0.0, 1.0)


def daily_risk(
    self_s: Any, iso_s: Any, peer_s: Any, weights: FusionWeights
) -> Any:
    """Fuse the three ``0``–``1`` scores into a ``Daily_Risk`` in ``0``–``100``.

    Computes ``100 * (w_self*self + w_iso*iso + w_peer*peer)`` using the fusion
    weights carried on ``weights`` (``config.weights`` — read from the config,
    never hard-coded), exactly as the design specifies (Property 15, Req
    6.4/6.5). With the default weights a day scoring ``self=0.8, iso=0.6,
    peer=0.4`` yields ``66.0``.

    The three score arguments may each be a scalar or an array/Series; scalars
    return a Python ``float`` and array-like inputs return a float64 ``ndarray``
    computed element-wise. When all inputs lie in ``[0, 1]`` and the weights sum
    to ``1`` (both guaranteed by the config loader and the upstream scorers),
    the result lies in ``[0, 100]``.
    """
    self_a = np.asarray(self_s, dtype="float64")
    iso_a = np.asarray(iso_s, dtype="float64")
    peer_a = np.asarray(peer_s, dtype="float64")

    risk = 100.0 * (
        weights.self_w * self_a + weights.iso_w * iso_a + weights.peer_w * peer_a
    )
    if risk.ndim == 0:
        return float(risk)
    return risk


def weekly_risk(daily_series: Any) -> pd.Series:
    """Roll a single user's daily risks into trailing-7 weekly risks.

    ``daily_series`` is one user's :func:`daily_risk` values ordered by day
    (a Series or any array-like). Returns a Series, aligned to the input, where
    each element is the sum of that value and up to the six preceding values —
    the trailing seven rows (fewer near the start). This is the per-row
    ``Weekly_Risk`` column stored in ``risk.parquet``; its final element is the
    sum of the user's last seven daily risks (design Property 16, Req 6.6).

    The window is measured in *rows* ordered by day, not calendar days, so a gap
    in a user's active days never dilutes the sum — see the module docstring.
    """
    series = daily_series if isinstance(daily_series, pd.Series) else pd.Series(
        list(daily_series), dtype="float64"
    )
    return series.rolling(window=WEEKLY_WINDOW, min_periods=1).sum()


def reason_record(user_day: Any, baselines: Any) -> list[dict]:
    """Return a user-day's Reason_Record: its top-3 self-anomaly features.

    A Reason_Record explains *why a day looked risky* by listing the (up to)
    three features that stood out most against the user's **own** baseline —
    each with its raw ``value``, the ``baseline_mean`` it was compared against,
    and its capped ``z_score`` (design Property 17, Req 6.7). The design's
    Scoring Algorithm draws the reasons from the self capped z-scores, so this
    reuses the Statistical_Scorer's :func:`self_score`, which already returns the
    top-3 features sorted by descending z-score.

    Parameters
    ----------
    user_day:
        One user-day row (dict, pandas Series, or one-row DataFrame) carrying a
        ``user`` field and the feature columns.
    baselines:
        The self :class:`~src.ueba.baselines.builder.Baseline` (per-user mean and
        std). A ``(self_baseline, peer_baseline, ...)`` tuple is also accepted
        for convenience — its first element is used — since reasons come from the
        self comparison only.

    Returns a list of at most :data:`~src.ueba.scoring.statistical.TOP_K` dicts;
    an empty list when the user has no baseline or no comparable features.
    """
    self_baseline = baselines[0] if isinstance(baselines, tuple) else baselines
    _score, top_features = self_score(user_day, self_baseline)
    return top_features[:TOP_K]


def compute_risk(
    user_day_features: pd.DataFrame,
    config: Config,
    model: Optional[IsolationModel] = None,
    roles: "pd.Series | dict | None" = None,
) -> pd.DataFrame:
    """Fuse baselines, statistical scores, and isolation scores into risk.

    This is the Scoring_Engine's end-to-end driver. Given the full per-user-day
    feature table it:

    1. builds the self and peer baselines (task 3.3) from the feature table;
    2. computes every user-day's ``self_score`` and ``peer_score`` plus the
       self/peer top-feature records (task 3.5);
    3. trains an :class:`IsolationModel` on the baseline period if ``model`` is
       ``None`` (otherwise scores with the one supplied), scores every user-day,
       and scales the raw scores into the ``0``–``1`` ``isolation_score`` via
       :func:`scale_isolation`;
    4. fuses the three scores into ``daily_risk`` with :func:`daily_risk` using
       the configured weights;
    5. rolls each user's daily risks into ``weekly_risk`` with
       :func:`weekly_risk` (trailing-7, per user, ordered by day); and
    6. attaches the Reason_Record (the self top-3 features) as ``reasons``.

    Returns a DataFrame with the columns of ``risk.parquet`` in design order
    (see :data:`RISK_COLUMNS`). The model is trained here only when not
    provided; persisting it is task 4.1's :func:`~src.ueba.models.save`.

    Roles for the peer baseline are taken from ``roles`` when given, otherwise
    from the ``role`` column already on ``user_day_features``.
    """
    self_baseline, peer_baseline, _meta = build_baselines(
        user_day_features, config, roles=roles
    )

    # Statistical scores, in the same row order as user_day_features.
    stat = score_statistical(user_day_features, self_baseline, peer_baseline)

    # Isolation scores. Train on the baseline period if no model was supplied.
    iso_model = (
        model if model is not None else train_from_features(user_day_features, config)
    )
    raw_iso = iso_score_model(iso_model, user_day_features)
    stat["isolation_score"] = scale_isolation(raw_iso)

    # Defensive fill: a user/role absent from its baseline scores NaN; such a
    # row cannot be fused, so it contributes 0 on that axis rather than poisoning
    # the arithmetic. In practice every user appears in the self baseline (it is
    # built from these same rows), so this only guards genuinely unscorable rows.
    for col in ("self_score", "peer_score", "isolation_score"):
        stat[col] = stat[col].fillna(0.0)

    # Fuse into Daily_Risk (0-100) with the configured weights.
    stat["daily_risk"] = daily_risk(
        stat["self_score"], stat["isolation_score"], stat["peer_score"], config.weights
    )

    # Order by (user, day) so the trailing-7 window walks each user's days in
    # time order, then roll into Weekly_Risk per user.
    stat = stat.sort_values(["user", "day"]).reset_index(drop=True)
    stat["weekly_risk"] = stat.groupby("user", sort=False)["daily_risk"].transform(
        weekly_risk
    )

    # Reason_Record: the self top-3 features already computed by the scorer,
    # sorted by descending z-score, capped at TOP_K.
    stat["reasons"] = stat["self_top_features"].apply(lambda feats: list(feats)[:TOP_K])

    result = stat[RISK_COLUMNS].copy()
    result["user"] = result["user"].astype("string")
    result["day"] = pd.to_datetime(result["day"])
    return result


def write_risk(df: pd.DataFrame, config: Config) -> Path:
    """Write the fused risk frame to ``<processed_dir>/risk.parquet``.

    Uses an explicit Arrow schema so the ``reasons`` list-of-struct column
    persists and reads back exactly, even for rows whose reason list is empty
    (design Property 5 round-trip, Req 6.8). Returns the output path.
    """
    processed_dir = Path(config.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / RISK_FILENAME

    table = pa.Table.from_pandas(df, schema=_RISK_SCHEMA, preserve_index=False)
    pq.write_table(table, out_path)
    return out_path
