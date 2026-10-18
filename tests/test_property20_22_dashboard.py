"""Property-based tests for the Streamlit-free dashboard logic.

Feature: ueba-insider-threat-detection, Property 20: Users are ranked by
descending weekly risk.

Feature: ueba-insider-threat-detection, Property 21: Drill-down data matches
stored data.

Feature: ueba-insider-threat-detection, Property 22: Alerts-per-day matches the
threshold definition.

These exercise the pure helpers in ``src/ueba/dashboard/logic.py`` (exported
from the ``ueba.dashboard`` package) directly on synthetic, in-memory risk
frames built by Hypothesis. No Streamlit runtime, no ``app.py`` import, and no
real pipeline artifacts are needed: every example constructs its own DataFrame
with the columns the dashboard reads (``user``, ``day``, ``daily_risk``,
``weekly_risk``, ``reasons``) and checks each helper against an independently
computed reference.
"""

from __future__ import annotations

import math

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ueba.dashboard import (
    alerts_per_day,
    rank_users,
    user_daily_timeline,
    user_peak_day,
    user_reasons,
)

# The fields the Scoring_Engine stores in each Reason_Record entry, derived
# here independently of the implementation so the comparison is a genuine check.
_REASON_FIELDS = ("feature", "value", "baseline_mean", "z_score")

# A small pool of feature names so planted reasons look like real contributors.
_FEATURE_NAMES = (
    "usb_connects",
    "after_hours_logons",
    "http_volume",
    "external_email",
    "logon_count",
)

# All synthetic days hang off this anchor so calendar days line up across users
# (needed so alerts-per-day can aggregate several users onto the same day).
_ANCHOR_DAY = pd.Timestamp("2010-01-04")


def _finite_float(min_value: float, max_value: float) -> st.SearchStrategy:
    """A plain finite float in ``[min_value, max_value]`` (no NaN/inf)."""
    return st.floats(
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )


@st.composite
def _reason_list(draw) -> list[dict]:
    """Draw 0–3 Reason_Record dicts carrying exactly the stored fields."""
    n = draw(st.integers(min_value=0, max_value=3))
    reasons: list[dict] = []
    for _ in range(n):
        reasons.append(
            {
                "feature": draw(st.sampled_from(_FEATURE_NAMES)),
                "value": draw(_finite_float(0.0, 1000.0)),
                "baseline_mean": draw(_finite_float(0.0, 1000.0)),
                "z_score": draw(_finite_float(-5.0, 5.0)),
            }
        )
    return reasons


@st.composite
def _risk_frames(draw):
    """Build a synthetic risk frame plus selections used by the properties.

    Returns a tuple ``(df, sel_user, sel_day, boundary_threshold,
    free_threshold)`` where:

    * ``df`` has one row per user-day with columns ``user``, ``day`` (a midnight
      Timestamp), ``daily_risk``, ``weekly_risk``, and ``reasons`` (a list of
      dicts). Each user's days are distinct (so a user-day is unique), but
      different users may share a calendar day.
    * ``sel_user`` is a user present in the frame and ``sel_day`` is one of that
      user's days — used to exercise the drill-down helpers.
    * ``boundary_threshold`` equals some existing ``daily_risk`` value so the
      inclusive ``>=`` boundary of alerts-per-day is exercised; ``free_threshold``
      is an arbitrary threshold.
    """
    n_users = draw(st.integers(min_value=1, max_value=5))
    users = [f"USER{i:04d}" for i in range(1, n_users + 1)]

    pool_size = draw(st.integers(min_value=1, max_value=12))
    day_pool = [_ANCHOR_DAY + pd.Timedelta(days=d) for d in range(pool_size)]

    rows: list[dict] = []
    for user in users:
        # Distinct days for this user (a user-day must be unique).
        user_days = draw(
            st.lists(
                st.sampled_from(day_pool),
                min_size=1,
                max_size=len(day_pool),
                unique=True,
            )
        )
        for day in user_days:
            rows.append(
                {
                    "user": user,
                    "day": day,
                    "daily_risk": draw(_finite_float(0.0, 100.0)),
                    "weekly_risk": draw(_finite_float(0.0, 700.0)),
                    "reasons": draw(_reason_list()),
                }
            )

    df = pd.DataFrame(rows, columns=["user", "day", "daily_risk", "weekly_risk", "reasons"])

    # A user present in the frame, and one of that user's actual days.
    sel_user = draw(st.sampled_from(users))
    sel_day = draw(st.sampled_from(sorted(df.loc[df["user"] == sel_user, "day"].tolist())))

    # An existing daily_risk value to pin the inclusive >= boundary, plus a free one.
    boundary_threshold = draw(st.sampled_from(df["daily_risk"].tolist()))
    free_threshold = draw(_finite_float(0.0, 100.0))

    return df, sel_user, sel_day, boundary_threshold, free_threshold


# ---------------------------------------------------------------------------
# Property 20 — ranking by descending weekly risk
# ---------------------------------------------------------------------------


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(payload=_risk_frames())
def test_users_ranked_by_descending_weekly_risk(payload):
    """Property 20: ranking orders users by non-increasing Weekly_Risk.

    Feature: ueba-insider-threat-detection, Property 20: Users are ranked by
    descending weekly risk. Validates Requirements 8.1.

    ``rank_users`` must return one row per user whose ``weekly_risk`` is that
    user's maximum in the input, ordered so the column is non-increasing from
    top to bottom, with ties broken by ``user`` ascending.
    """
    df = payload[0]
    result = rank_users(df)

    # Independently computed reference: per-user peak weekly_risk, then sorted
    # by weekly_risk descending with ties broken by user ascending.
    peak: dict[str, float] = {}
    for user, weekly in zip(df["user"], df["weekly_risk"]):
        if user not in peak or weekly > peak[user]:
            peak[user] = weekly
    expected = sorted(peak.items(), key=lambda kv: (-kv[1], kv[0]))
    expected_users = [u for u, _ in expected]
    expected_weekly = [w for _, w in expected]

    result_users = [str(u) for u in result["user"].tolist()]
    result_weekly = result["weekly_risk"].tolist()

    # One row per user, each equal to that user's max weekly_risk, in order.
    assert result_users == expected_users
    assert result_weekly == expected_weekly
    assert len(result) == df["user"].nunique()

    # The weekly_risk column is non-increasing from top to bottom.
    for earlier, later in zip(result_weekly, result_weekly[1:]):
        assert earlier >= later

    # Among equal weekly_risk values, users are in ascending order.
    for i in range(len(result_users) - 1):
        if result_weekly[i] == result_weekly[i + 1]:
            assert result_users[i] < result_users[i + 1]


# ---------------------------------------------------------------------------
# Property 21 — drill-down data matches stored data
# ---------------------------------------------------------------------------


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(payload=_risk_frames())
def test_drilldown_matches_stored_data(payload):
    """Property 21: timeline and why-flagged panel equal stored values.

    Feature: ueba-insider-threat-detection, Property 21: Drill-down data matches
    stored data. Validates Requirements 8.3, 8.4, 8.7.

    For a selected user, ``user_daily_timeline`` must equal that user's
    ``(day, daily_risk)`` rows in date order, ``user_peak_day`` the earliest day
    of their maximum Daily_Risk, and ``user_reasons`` the stored Reason_Record
    fields for the chosen user-day.
    """
    df, sel_user, sel_day, _boundary, _free = payload

    # --- timeline equals the user's stored (day, daily_risk) in date order ---
    timeline = user_daily_timeline(df, sel_user)

    user_rows = df.loc[df["user"] == sel_user, ["day", "daily_risk"]]
    user_rows = user_rows.sort_values(by="day").reset_index(drop=True)
    expected_days = [pd.Timestamp(d) for d in user_rows["day"].tolist()]
    expected_risks = user_rows["daily_risk"].tolist()

    assert [pd.Timestamp(d) for d in timeline["day"].tolist()] == expected_days
    assert timeline["daily_risk"].tolist() == expected_risks

    # --- peak day is the earliest day achieving the user's max Daily_Risk ---
    max_risk = max(expected_risks)
    expected_peak = min(
        d for d, r in zip(expected_days, expected_risks) if r == max_risk
    )
    assert user_peak_day(df, sel_user) == expected_peak

    # --- why-flagged panel equals the stored Reason_Record for that user-day ---
    stored = df.loc[
        (df["user"] == sel_user) & (df["day"] == sel_day), "reasons"
    ].iloc[0]
    expected_reasons = [
        {field: entry[field] for field in _REASON_FIELDS} for entry in stored
    ]
    assert user_reasons(df, sel_user, sel_day) == expected_reasons


# ---------------------------------------------------------------------------
# Property 22 — alerts-per-day matches the threshold definition
# ---------------------------------------------------------------------------


def _expected_alerts_per_day(df: pd.DataFrame, threshold: float):
    """Reference alerts-per-day: user-days with daily_risk >= threshold."""
    counts: dict[pd.Timestamp, int] = {}
    for day, risk in zip(df["day"], df["daily_risk"]):
        if risk >= threshold:
            key = pd.Timestamp(day).normalize()
            counts[key] = counts.get(key, 0) + 1
    items = sorted(counts.items(), key=lambda kv: kv[0])
    return [d for d, _ in items], [c for _, c in items]


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(payload=_risk_frames())
def test_alerts_per_day_matches_threshold(payload):
    """Property 22: per-day alert counts equal user-days at/above threshold.

    Feature: ueba-insider-threat-detection, Property 22: Alerts-per-day matches
    the threshold definition. Validates Requirements 8.5.

    For any threshold, ``alerts_per_day`` returns one row per day with at least
    one alert, ordered by day, whose count equals the number of user-days that
    day with ``daily_risk`` at or above the threshold (inclusive ``>=``).
    """
    df, _sel_user, _sel_day, boundary_threshold, free_threshold = payload

    # Arbitrary threshold.
    result = alerts_per_day(df, free_threshold)
    exp_days, exp_counts = _expected_alerts_per_day(df, free_threshold)
    assert [pd.Timestamp(d) for d in result["day"].tolist()] == exp_days
    assert result["alert_count"].tolist() == exp_counts

    # Inclusive >= boundary: a threshold equal to an existing daily_risk value
    # must count that user-day, so the result cannot be empty, and it still
    # matches the independently computed reference exactly.
    boundary_result = alerts_per_day(df, boundary_threshold)
    b_days, b_counts = _expected_alerts_per_day(df, boundary_threshold)
    assert not boundary_result.empty
    assert [pd.Timestamp(d) for d in boundary_result["day"].tolist()] == b_days
    assert boundary_result["alert_count"].tolist() == b_counts

    # The total alert count never exceeds the number of user-days.
    assert boundary_result["alert_count"].sum() <= len(df)
