"""Baseline_Builder — learn what "normal" looks like for each person and role.

Anomaly detection only means something if we compare a day against a stable
picture of ordinary behaviour. This module builds that picture. For every user
it picks an early slice of their history — the *Baseline_Period* — and summarises
each behavioural feature over that slice as a mean and a standard deviation. It
does the same per job-role so a user-day can later be compared against both the
user's *own* normal (the self baseline) and their *peers'* normal (the peer
baseline).

Three rules from the design drive everything here:

* **Split by time, never randomly.** The baseline uses only user-days that fall
  inside the Baseline_Period. Out-of-period rows never touch a baseline
  (design Property 10).
* **Respect the window bounds.** The period starts at the user's first active
  day and spans the configured target length, clamped to the 30–90 day bounds
  from ``config.baseline`` (design Property 11).
* **Handle the awkward cases.** A user with too little history is *flagged
  insufficient* so scoring can skip them rather than trust an unstable baseline.
  A feature whose baseline never varies gets a standard deviation of ``0.0``;
  downstream the z-score for such a feature is defined as ``0`` (no deviation
  signal, and no division by zero).

What this module does **not** do: it does not compute z-scores or risk. It only
produces the means, standard deviations, day-counts, and sufficiency flags that
the Statistical_Scorer (task 3.5) consumes.

Cutoff convention
-----------------
``baseline_cutoff`` returns the **last day included** in the Baseline_Period
(an *inclusive* boundary). A user-day is in-period when
``first_day <= day <= cutoff``. The span from ``first_day`` to ``cutoff``
inclusive is exactly the (clamped) target number of calendar days.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import BaselineConfig, Config
from ..features import FEATURE_COLUMNS

__all__ = [
    "Baseline",
    "baseline_cutoff",
    "split_by_cutoff",
    "select_baseline_rows",
    "build_self_baseline",
    "build_peer_baseline",
    "build_baselines",
]


@dataclass
class Baseline:
    """A set of per-group feature baselines (one group = one user or one role).

    Attributes
    ----------
    mean:
        DataFrame indexed by the group key (user id or role), one column per
        feature in :data:`FEATURE_COLUMNS`, holding the in-period mean.
    std:
        Same shape as ``mean`` but holding the in-period standard deviation.
        Zero-variance features are stored as ``0.0`` (never ``NaN``), so the
        scorer can safely define their z-score as ``0``.
    n_days:
        Series indexed by the group key giving how many in-period user-days
        contributed to that group's baseline.
    sufficient:
        Series indexed by the group key marking whether the group has enough
        history to trust (``n_days >= min_days``). ``None`` for peer baselines,
        where sufficiency is a per-user concept rather than a per-role one.
    """

    mean: pd.DataFrame
    std: pd.DataFrame
    n_days: pd.Series
    sufficient: "pd.Series | None" = None


def _effective_window_days(baseline: BaselineConfig) -> int:
    """Return the Baseline_Period length in days, clamped to the config bounds.

    The target length is pulled toward the configured ``target_days`` but is
    never allowed below ``min_days`` (30) or above ``max_days`` (90). This is
    the single place the 30–90 day rule is enforced.
    """
    span = baseline.target_days
    span = max(span, baseline.min_days)
    span = min(span, baseline.max_days)
    return span


def baseline_cutoff(user_days: pd.DataFrame, config: Config) -> pd.Timestamp:
    """Return the last day included in one user's Baseline_Period.

    ``user_days`` holds the user-day rows for a *single* user and must contain a
    ``day`` column. The period starts at that user's first active day and spans
    the configured target length clamped to the 30–90 day bounds. The returned
    cutoff is the *inclusive* last day: a row is in-period when
    ``first_day <= day <= cutoff``.

    Raises
    ------
    ValueError
        If ``user_days`` is empty (no first day to anchor the window on).
    """
    if user_days.empty:
        raise ValueError("Cannot compute a baseline cutoff for an empty history.")

    first_day = pd.Timestamp(user_days["day"].min()).normalize()
    span = _effective_window_days(config.baseline)
    # span days inclusive of the first day -> add (span - 1) days.
    return first_day + pd.Timedelta(days=span - 1)


def split_by_cutoff(
    user_days: pd.DataFrame, cutoff: pd.Timestamp
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Split one user's rows into (in-period, later) at an inclusive cutoff.

    Rows on or before ``cutoff`` form the baseline period; rows strictly after
    it belong to the scoring period. The ``day`` column is compared as a
    timestamp so date/datetime inputs behave identically.
    """
    days = pd.to_datetime(user_days["day"])
    cutoff = pd.Timestamp(cutoff)
    in_period = user_days[days <= cutoff]
    later = user_days[days > cutoff]
    return in_period, later


def select_baseline_rows(
    user_day_features: pd.DataFrame, config: Config
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Pick each user's in-period rows and describe their baseline window.

    Walks every user in ``user_day_features`` (which must have ``user`` and
    ``day`` columns), computes that user's cutoff, and keeps only the rows that
    fall inside the Baseline_Period. It returns two frames:

    * the concatenation of all users' in-period rows (ready for
      :func:`build_self_baseline` / :func:`build_peer_baseline`), and
    * a per-user *meta* frame indexed by user with ``first_day``, ``cutoff``,
      ``n_days`` (distinct active days in-period), and ``sufficient`` (whether
      ``n_days >= config.baseline.min_days``).

    Users flagged ``sufficient == False`` have too little early history to build
    a trustworthy baseline and should be excluded from scoring.
    """
    min_days = config.baseline.min_days
    in_period_parts: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    for user, group in user_day_features.groupby("user", sort=False):
        cutoff = baseline_cutoff(group, config)
        first_day = pd.Timestamp(group["day"].min()).normalize()
        in_period, _later = split_by_cutoff(group, cutoff)
        n_days = int(pd.to_datetime(in_period["day"]).dt.normalize().nunique())
        in_period_parts.append(in_period)
        meta_rows.append(
            {
                "user": user,
                "first_day": first_day,
                "cutoff": cutoff,
                "n_days": n_days,
                "sufficient": n_days >= min_days,
            }
        )

    if in_period_parts:
        in_period_all = pd.concat(in_period_parts, ignore_index=True)
    else:
        in_period_all = user_day_features.iloc[0:0].copy()

    meta = pd.DataFrame(
        meta_rows, columns=["user", "first_day", "cutoff", "n_days", "sufficient"]
    ).set_index("user")
    return in_period_all, meta


def _group_mean_std(
    frame: pd.DataFrame, group_col: str
) -> "tuple[pd.DataFrame, pd.DataFrame, pd.Series]":
    """Compute per-group mean, std (ddof=0), and row count over the features.

    Standard deviation uses ``ddof=0`` (population) so a group with a single
    row, or a feature that never varies, yields ``0.0`` rather than ``NaN`` —
    the zero-variance convention the design requires. Any residual ``NaN`` (for
    an empty group) is filled with ``0.0`` as a final guard.
    """
    features = [c for c in FEATURE_COLUMNS if c in frame.columns]
    grouped = frame.groupby(group_col, sort=True)
    mean = grouped[features].mean()
    std = grouped[features].std(ddof=0).fillna(0.0)
    n_days = grouped.size().rename("n_days")
    return mean, std, n_days


def build_self_baseline(
    features_in_period: pd.DataFrame, min_days: "int | None" = None
) -> Baseline:
    """Build each user's own baseline from their in-period user-days.

    ``features_in_period`` must already be restricted to Baseline_Period rows
    (see :func:`select_baseline_rows`) and must carry a ``user`` column plus the
    feature columns. Returns a :class:`Baseline` whose ``mean`` and ``std`` are
    indexed by user. When ``min_days`` is given, each user is also marked
    ``sufficient`` when they have at least that many in-period days, so scoring
    can drop short-history users.
    """
    mean, std, n_days = _group_mean_std(features_in_period, "user")
    sufficient = None
    if min_days is not None:
        sufficient = (n_days >= min_days).rename("sufficient")
    return Baseline(mean=mean, std=std, n_days=n_days, sufficient=sufficient)


def build_peer_baseline(
    features_in_period: pd.DataFrame, roles: "pd.Series | dict | None" = None
) -> Baseline:
    """Build each role group's baseline from its members' in-period user-days.

    ``features_in_period`` must already be restricted to Baseline_Period rows.
    The role of each row is taken from ``roles`` when provided (a mapping or
    Series of ``user -> role``); otherwise the frame's existing ``role`` column
    is used. Rows whose role is unknown/null are dropped before grouping.
    Returns a :class:`Baseline` whose ``mean`` and ``std`` are indexed by role.
    """
    frame = features_in_period.copy()

    if roles is not None:
        mapping = dict(roles) if not isinstance(roles, dict) else roles
        frame["role"] = frame["user"].map(mapping)
    elif "role" not in frame.columns:
        raise ValueError(
            "build_peer_baseline needs role information: pass `roles` or include "
            "a 'role' column in features_in_period."
        )

    frame = frame[frame["role"].notna()]
    mean, std, n_days = _group_mean_std(frame, "role")
    return Baseline(mean=mean, std=std, n_days=n_days, sufficient=None)


def build_baselines(
    user_day_features: pd.DataFrame,
    config: Config,
    roles: "pd.Series | dict | None" = None,
) -> "tuple[Baseline, Baseline, pd.DataFrame]":
    """Build self and peer baselines end-to-end from a full user-day feature set.

    This is the convenience entry point: it selects each user's in-period rows
    (honouring the per-user cutoff), then builds the per-user self baseline and
    the per-role peer baseline from exactly those rows. It returns
    ``(self_baseline, peer_baseline, meta)`` where ``meta`` is the per-user
    window description from :func:`select_baseline_rows` (including the
    ``sufficient`` flag used to exclude short-history users from scoring).
    """
    in_period, meta = select_baseline_rows(user_day_features, config)
    self_baseline = build_self_baseline(in_period, min_days=config.baseline.min_days)
    peer_baseline = build_peer_baseline(in_period, roles=roles)
    return self_baseline, peer_baseline, meta
