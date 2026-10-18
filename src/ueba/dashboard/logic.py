"""Pure, Streamlit-free logic behind the dashboard (ranking + controls).

This module is deliberately free of any ``import streamlit``. Everything here is
an ordinary function over a pandas DataFrame, so the Phase 6 property tests
(Property 20 — ranking by descending Weekly_Risk; and the risk-threshold filter)
can be checked headlessly, and so ``app.py`` stays a thin rendering shell on top
of logic that is easy to reason about and test.

Plain-English tour:

* :func:`risk_parquet_path` — where the dashboard expects ``risk.parquet`` to
  live, derived from the config's ``processed_dir`` (never hard-coded).
* :func:`load_risk` — read that Parquet file into a DataFrame, or raise
  :class:`FileNotFoundError` with an actionable message when it is absent so the
  caller can show a friendly error instead of crashing (Req 8.1 graceful start).
* :func:`rank_users` — collapse the per-user-day risk frame to one row per user
  (their peak Weekly_Risk) ordered most-risky-first, matching the evaluation
  harness's :func:`rank_users_by_weekly_risk` exactly (Req 8.1, Property 20).
* :func:`filter_by_threshold` — keep only the user-days whose risk is at or
  above a chosen threshold, powering the risk-threshold control (Req 8.2).
* :func:`available_day_bounds` — the earliest and latest day present in the
  data, used to seed the date control (Req 8.2).

Phase 6.2 adds the per-user drill-down and the alerts-per-day helpers:

* :func:`user_daily_timeline` — one user's Daily_Risk over time, in date order,
  feeding the drill-down timeline chart (Req 8.3, Property 21).
* :func:`user_peak_day` — the single day a user's Daily_Risk peaked, a sensible
  default day to explain in the why-flagged panel.
* :func:`user_reasons` — the stored Reason_Record (top contributing features,
  each with its value, baseline mean, and z-score) for one user-day, powering the
  why-flagged panel and the "top reasons" for known insiders (Req 8.4, 8.7,
  Property 21).
* :func:`alerts_per_day` — how many user-days each day sit at or above the
  threshold, feeding the alerts-per-day view (Req 8.5, Property 22).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import Config
from ..scoring import RISK_FILENAME

__all__ = [
    "RISK_PARQUET_MISSING_MESSAGE",
    "risk_parquet_path",
    "load_risk",
    "rank_users",
    "filter_by_threshold",
    "available_day_bounds",
    "user_daily_timeline",
    "user_peak_day",
    "user_reasons",
    "alerts_per_day",
]

# The column the ranking collapses to one-per-user — a user's peak Weekly_Risk.
# Kept identical to the evaluation harness so the dashboard and the metrics agree
# on who the "top" users are.
_WEEKLY_RISK_COLUMN = "weekly_risk"

# The per-user-day column the risk-threshold control filters on. Daily_Risk is
# the 0-100 fused score for a single day, which is the natural unit for "which
# user-days are alerts today".
_DAILY_RISK_COLUMN = "daily_risk"

# Shown to the analyst (and returned by load_risk's error) when the pipeline has
# not produced risk.parquet yet. Phrased as a concrete next step, not a stack
# trace, per Req 8.1 ("actionable message ... rather than crashing").
RISK_PARQUET_MISSING_MESSAGE = (
    "risk.parquet not found — the dashboard reads a pre-computed risk file. "
    "Generate it first by running the scoring pipeline "
    "(for example `python scripts/run_eval.py`, or the ingestion → scoring "
    "steps), then reload this page."
)


def risk_parquet_path(config: Config) -> Path:
    """Return the expected location of ``risk.parquet`` for this config.

    Resolved from ``config.processed_dir`` and the scoring layer's
    :data:`~src.ueba.scoring.RISK_FILENAME`, so the dashboard looks in exactly
    the place the Scoring_Engine writes to — no hard-coded path.
    """
    return Path(config.processed_dir) / RISK_FILENAME


def load_risk(config: Config) -> pd.DataFrame:
    """Load the pre-computed ``risk.parquet`` into a DataFrame.

    The dashboard never recomputes risk at render time; it reads this one
    artifact (design: "reads pre-computed ``risk.parquet``, no heavy compute at
    render time"). The ``day`` column is coerced to pandas datetime so the date
    control and timeline code can rely on it.

    Raises :class:`FileNotFoundError` carrying
    :data:`RISK_PARQUET_MISSING_MESSAGE` when the file is absent, so the Streamlit
    layer can surface an actionable message and stop gracefully rather than
    letting an unhandled exception crash the app (Req 8.1).
    """
    path = risk_parquet_path(config)
    if not path.is_file():
        raise FileNotFoundError(RISK_PARQUET_MISSING_MESSAGE)

    df = pd.read_parquet(path)
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"])
    return df


def rank_users(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Rank users most-risky-first by their peak Weekly_Risk.

    ``Weekly_Risk`` is stored per user-day, so a user is represented by their
    **maximum** Weekly_Risk across all their days — the same rule the evaluation
    harness uses in :func:`~src.ueba.evaluation.metrics.rank_users_by_weekly_risk`,
    so the dashboard's "top users" match the metrics' ranking. Rows are sorted by
    that peak descending; ties are broken by ``user`` ascending so the ordering
    is deterministic (Req 8.1, Property 20 — Weekly_Risk is non-increasing from
    top to bottom).

    Returns a two-column DataFrame (``user``, ``weekly_risk``) with one row per
    user, index reset. An empty input yields an empty frame with those columns.
    """
    if risk_df.empty:
        return pd.DataFrame({"user": pd.Series(dtype="string"),
                             _WEEKLY_RISK_COLUMN: pd.Series(dtype="float64")})

    per_user = (
        risk_df.groupby(risk_df["user"].astype("string"))[_WEEKLY_RISK_COLUMN]
        .max()
        .reset_index()
    )
    per_user = per_user.sort_values(
        by=[_WEEKLY_RISK_COLUMN, "user"], ascending=[False, True]
    ).reset_index(drop=True)
    return per_user


def filter_by_threshold(
    risk_df: pd.DataFrame,
    threshold: float,
    column: str = _DAILY_RISK_COLUMN,
) -> pd.DataFrame:
    """Keep only the user-days whose risk is at or above ``threshold``.

    Powers the risk-threshold control: an analyst drags the threshold and sees
    only the user-days that qualify as alerts. The comparison is inclusive
    (``>= threshold``) so a user-day sitting exactly on the threshold counts as
    an alert, matching the alerts-per-day definition used later in Phase 6
    (Property 22). Filters on ``daily_risk`` by default; ``column`` lets a caller
    threshold on ``weekly_risk`` instead.

    Returns a copy of the matching rows with the original index reset. An empty
    input (or one missing ``column``) yields an empty frame.
    """
    if risk_df.empty or column not in risk_df.columns:
        return risk_df.iloc[0:0].copy()
    mask = risk_df[column] >= threshold
    return risk_df.loc[mask].reset_index(drop=True)


def available_day_bounds(
    risk_df: pd.DataFrame,
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return the ``(earliest, latest)`` day present in the risk frame.

    Used to seed the date control's range so the analyst can only pick days the
    data actually covers. Returns ``(None, None)`` when the frame is empty or has
    no ``day`` column.
    """
    if risk_df.empty or "day" not in risk_df.columns:
        return (None, None)
    days = pd.to_datetime(risk_df["day"])
    return (days.min(), days.max())


# Column name for the alerts-per-day count, kept descriptive so the chart and
# any test refer to it by the same key.
_ALERT_COUNT_COLUMN = "alert_count"

# The fields the Scoring_Engine stores in each Reason_Record entry (see
# ``_REASON_STRUCT`` in src/ueba/scoring/engine.py). Used to normalise the
# reasons back into plain dicts regardless of how Parquet re-materialised them.
_REASON_FIELDS = ("feature", "value", "baseline_mean", "z_score")


def user_daily_timeline(risk_df: pd.DataFrame, user: object) -> pd.DataFrame:
    """Return one user's Daily_Risk over time, ordered oldest day first.

    This is the data behind the drill-down timeline: pick a user, get back just
    their user-days as a two-column frame (``day``, ``daily_risk``) sorted by
    ``day`` ascending so the chart reads left-to-right in calendar order. The
    values are copied straight from the stored ``risk.parquet`` — the dashboard
    never recomputes risk at render time (Req 8.3, 8.6; Property 21: the timeline
    equals the user's stored Daily_Risk values in date order).

    An unknown user, an empty frame, or a frame missing the needed columns yields
    an empty ``(day, daily_risk)`` frame.
    """
    if (
        risk_df.empty
        or "user" not in risk_df.columns
        or "day" not in risk_df.columns
        or _DAILY_RISK_COLUMN not in risk_df.columns
    ):
        return pd.DataFrame(
            {
                "day": pd.Series(dtype="datetime64[ns]"),
                _DAILY_RISK_COLUMN: pd.Series(dtype="float64"),
            }
        )

    rows = risk_df.loc[risk_df["user"] == user, ["day", _DAILY_RISK_COLUMN]].copy()
    rows["day"] = pd.to_datetime(rows["day"])
    rows = rows.sort_values(by="day").reset_index(drop=True)
    return rows


def user_peak_day(risk_df: pd.DataFrame, user: object) -> Optional[pd.Timestamp]:
    """Return the day a user's Daily_Risk peaked, or ``None`` if they have none.

    A convenient default day to explain in the why-flagged panel: the single
    riskiest day is usually the one an analyst wants to understand first. Ties
    (two days with the same peak Daily_Risk) resolve to the earliest day so the
    choice is deterministic.
    """
    timeline = user_daily_timeline(risk_df, user)
    if timeline.empty:
        return None
    # idxmax picks the first occurrence of the maximum; the timeline is already
    # sorted ascending by day, so that first occurrence is the earliest peak day.
    peak_idx = timeline[_DAILY_RISK_COLUMN].idxmax()
    return pd.Timestamp(timeline.loc[peak_idx, "day"])


def user_reasons(
    risk_df: pd.DataFrame, user: object, day: object
) -> list[dict]:
    """Return the stored Reason_Record for one user-day as a list of dicts.

    The why-flagged panel answers "why was this user flagged on this day?" by
    showing the top contributing features the Scoring_Engine already stored — each
    with its ``feature`` name, its raw ``value``, the ``baseline_mean`` it was
    compared against, and its ``z_score`` (value vs baseline). This reads those
    stored reasons back verbatim; it does not recompute them (Req 8.4, 8.7;
    Property 21: the panel equals the stored Reason_Record fields). It works the
    same for known insiders, giving their top contributing reasons (Req 8.7).

    ``day`` is matched on the calendar day so a time-of-day component never
    prevents a match. Returns an empty list when the user-day is absent, the
    frame has no ``reasons`` column, or that user-day's reason list is empty.
    """
    if (
        risk_df.empty
        or "user" not in risk_df.columns
        or "day" not in risk_df.columns
        or "reasons" not in risk_df.columns
    ):
        return []

    target_day = pd.Timestamp(day).normalize()
    days = pd.to_datetime(risk_df["day"]).dt.normalize()
    match = risk_df.loc[(risk_df["user"] == user) & (days == target_day)]
    if match.empty:
        return []

    # One user-day is expected; if duplicates ever exist, explain the first.
    raw_reasons = match.iloc[0]["reasons"]
    return _normalise_reasons(raw_reasons)


def _normalise_reasons(raw_reasons: object) -> list[dict]:
    """Coerce a stored reasons cell into a plain ``list[dict]``.

    Parquet may hand the ``reasons`` cell back as a numpy array of dicts, a list
    of dicts, or ``None``. This flattens all of those into a list of ordinary
    dicts carrying exactly the Reason_Record fields, so callers (and tests) get a
    stable shape no matter how the artifact was re-materialised.
    """
    if raw_reasons is None:
        return []
    # A scalar NaN (e.g. a missing cell) is not iterable in a useful way.
    if isinstance(raw_reasons, float):
        return []

    normalised: list[dict] = []
    for entry in raw_reasons:
        if entry is None:
            continue
        normalised.append({field: entry[field] for field in _REASON_FIELDS})
    return normalised


def alerts_per_day(risk_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Count, per day, how many user-days are alerts at or above ``threshold``.

    An "alert" is a user-day whose ``daily_risk`` is at or above the analyst's
    threshold — the same inclusive ``>=`` rule :func:`filter_by_threshold` uses,
    so the two controls agree on what counts (Req 8.5; Property 22: the count per
    day equals the number of user-days that day with Daily_Risk ≥ threshold). The
    result is a two-column frame (``day``, ``alert_count``) with one row per day
    that has at least one alert, ordered oldest day first, ready to chart as a
    bar/line over time.

    An empty frame, or one missing the needed columns, yields an empty
    ``(day, alert_count)`` frame.
    """
    empty = pd.DataFrame(
        {
            "day": pd.Series(dtype="datetime64[ns]"),
            _ALERT_COUNT_COLUMN: pd.Series(dtype="int64"),
        }
    )
    if (
        risk_df.empty
        or "day" not in risk_df.columns
        or _DAILY_RISK_COLUMN not in risk_df.columns
    ):
        return empty

    alerts = risk_df.loc[risk_df[_DAILY_RISK_COLUMN] >= threshold].copy()
    if alerts.empty:
        return empty

    alerts["day"] = pd.to_datetime(alerts["day"]).dt.normalize()
    counts = (
        alerts.groupby("day").size().reset_index(name=_ALERT_COUNT_COLUMN)
    )
    counts = counts.sort_values(by="day").reset_index(drop=True)
    counts[_ALERT_COUNT_COLUMN] = counts[_ALERT_COUNT_COLUMN].astype("int64")
    return counts
