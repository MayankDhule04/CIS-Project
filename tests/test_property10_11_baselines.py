"""Property-based tests for the Baseline_Builder.

Feature: ueba-insider-threat-detection, Property 10: Baselines depend only on
in-period user-days.

Feature: ueba-insider-threat-detection, Property 11: Baseline window respects
bounds.

These tests build user-day feature frames directly in memory with Hypothesis
(no CSVs, no temp files) and drive the public baselines API
(``src.ueba.baselines``). The project ``config.yaml`` is loaded once at import
time; each example varies only the baseline-window bounds via
:func:`dataclasses.replace`, always keeping ``min_days <= target_days <=
max_days`` inside the 30–90 range Requirement 5.2 mandates.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import date

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.ueba.baselines import (
    baseline_cutoff,
    build_baselines,
    build_self_baseline,
    select_baseline_rows,
)
from src.ueba.config import BaselineConfig, load_config
from src.ueba.features import FEATURE_COLUMNS

# Repository root -> the config.yaml the system reads. Loaded once; every
# example clones it with new baseline bounds rather than re-reading the file.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_CONFIG = load_config(os.path.join(_REPO_ROOT, "config.yaml"))

# A single role is enough for these properties: the self baseline is per-user
# and the peer baseline is per-role, and both are exercised by one role group.
_ROLE = "ANALYST"


@st.composite
def _user_histories(draw):
    """Draw one user's full history plus the baseline config to score it with.

    The returned bundle guarantees the history straddles the Baseline_Period:
    it always includes the user's first active day (offset 0) and at least one
    day strictly past the cutoff, so both in-period and out-of-period rows
    exist. Feature values are arbitrary non-negative integers.
    """
    # Window bounds kept inside 30..90 with min <= target <= max, so the
    # clamped span equals target_days and never escapes Requirement 5.2's range.
    min_days = draw(st.integers(min_value=30, max_value=90))
    max_days = draw(st.integers(min_value=min_days, max_value=90))
    target_days = draw(st.integers(min_value=min_days, max_value=max_days))
    span = target_days  # == clamp(target, min, max) because min <= target <= max

    first = draw(st.dates(min_value=date(2008, 1, 1), max_value=date(2015, 12, 31)))

    # In-period day offsets: the first active day (anchor) plus a few others,
    # all strictly before the cutoff (cutoff = first + span - 1).
    extra_in = draw(
        st.lists(
            st.integers(min_value=1, max_value=span - 1),
            unique=True,
            max_size=8,
        )
    )
    in_offsets = sorted({0, *extra_in})

    # Out-of-period day offsets: at least one day on/after the cutoff+1.
    out_offsets = sorted(
        draw(
            st.lists(
                st.integers(min_value=span, max_value=span + 90),
                unique=True,
                min_size=1,
                max_size=8,
            )
        )
    )

    offsets = in_offsets + out_offsets
    feature_rows = draw(
        st.lists(
            st.lists(
                st.integers(min_value=0, max_value=200),
                min_size=len(FEATURE_COLUMNS),
                max_size=len(FEATURE_COLUMNS),
            ),
            min_size=len(offsets),
            max_size=len(offsets),
        )
    )

    first_ts = pd.Timestamp(first)
    records = []
    for offset, values in zip(offsets, feature_rows):
        record = {
            "user": "USER0001",
            "day": first_ts + pd.Timedelta(days=offset),
            "role": _ROLE,
        }
        record.update(dict(zip(FEATURE_COLUMNS, values)))
        records.append(record)

    frame = pd.DataFrame(records, columns=["user", "day", "role", *FEATURE_COLUMNS])
    config = replace(
        _BASE_CONFIG,
        baseline=BaselineConfig(min_days=min_days, target_days=target_days, max_days=max_days),
    )
    return {
        "frame": frame,
        "config": config,
        "first_day": first_ts.normalize(),
        "span": span,
        "min_days": min_days,
        "in_offsets": in_offsets,
    }


# ---------------------------------------------------------------------------
# Property 11 — baseline window respects bounds
# ---------------------------------------------------------------------------


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=_user_histories())
def test_baseline_window_respects_bounds(case):
    """Property 11: the Baseline_Period starts at first activity, span in 30..90.

    Feature: ueba-insider-threat-detection, Property 11: Baseline window
    respects bounds. Validates Requirements 5.2.

    For any user history the inclusive cutoff equals first_day + (span - 1)
    days, the span lies within 30 to 90 days, and the window anchors on the
    user's first active day.
    """
    frame = case["frame"]
    config = case["config"]
    first_day = case["first_day"]
    span = case["span"]

    cutoff = baseline_cutoff(frame, config)

    # Span is always within the 30..90 Requirement 5.2 band.
    assert 30 <= span <= 90

    # The window anchors on the user's first activity (the minimum day).
    observed_first = pd.Timestamp(frame["day"].min()).normalize()
    assert observed_first == first_day

    # Inclusive cutoff is exactly span calendar days from the first day.
    assert cutoff == first_day + pd.Timedelta(days=span - 1)
    assert (cutoff - first_day).days + 1 == span

    # The per-user meta agrees: first_day anchors the window and the in-period
    # day count is exactly the distinct in-period days we planted.
    _in_period, meta = select_baseline_rows(frame, config)
    user_meta = meta.loc["USER0001"]
    assert pd.Timestamp(user_meta["first_day"]).normalize() == first_day
    assert pd.Timestamp(user_meta["cutoff"]) == cutoff
    assert int(user_meta["n_days"]) == len(case["in_offsets"])


# ---------------------------------------------------------------------------
# Property 10 — baselines depend only on in-period user-days
# ---------------------------------------------------------------------------


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=_user_histories())
def test_baselines_depend_only_on_in_period_days(case):
    """Property 10: changing/removing out-of-period days leaves baselines intact.

    Feature: ueba-insider-threat-detection, Property 10: Baselines depend only
    on in-period user-days. Validates Requirements 5.1.

    Mutating or dropping user-days strictly after the cutoff never changes the
    self or peer baseline, while mutating an in-period day does change the self
    baseline — showing the baseline reflects in-period data and only that.
    """
    frame = case["frame"]
    config = case["config"]
    first_day = case["first_day"]

    cutoff = baseline_cutoff(frame, config)
    out_mask = pd.to_datetime(frame["day"]) > cutoff

    self_full, peer_full, _meta_full = build_baselines(frame, config)

    # (a) Mutating out-of-period rows must leave both baselines untouched.
    mutated_out = frame.copy()
    mutated_out.loc[out_mask, FEATURE_COLUMNS] += 9999
    self_mut, peer_mut, _ = build_baselines(mutated_out, config)
    assert_frame_equal(self_full.mean, self_mut.mean)
    assert_frame_equal(self_full.std, self_mut.std)
    assert_frame_equal(peer_full.mean, peer_mut.mean)
    assert_frame_equal(peer_full.std, peer_mut.std)
    assert_series_equal(self_full.n_days, self_mut.n_days)

    # (b) Dropping out-of-period rows entirely must also leave them untouched.
    dropped_out = frame.loc[~out_mask].copy()
    self_drop, peer_drop, _ = build_baselines(dropped_out, config)
    assert_frame_equal(self_full.mean, self_drop.mean)
    assert_frame_equal(self_full.std, self_drop.std)
    assert_frame_equal(peer_full.mean, peer_drop.mean)
    assert_frame_equal(peer_full.std, peer_drop.std)

    # (c) Mutating an in-period row (the first active day) must change the self
    # baseline — the baseline genuinely reflects in-period user-days.
    mutated_in = frame.copy()
    in_anchor = pd.to_datetime(mutated_in["day"]).dt.normalize() == first_day
    feature = FEATURE_COLUMNS[0]
    mutated_in.loc[in_anchor, feature] += 1000
    self_in = build_self_baseline(
        select_baseline_rows(mutated_in, config)[0], min_days=config.baseline.min_days
    )
    assert not self_full.mean.equals(self_in.mean), (
        "self baseline mean did not change after mutating an in-period day"
    )
