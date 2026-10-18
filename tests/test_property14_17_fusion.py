"""Property-based tests for the Scoring_Engine's fusion stage.

Feature: ueba-insider-threat-detection, Property 14: Isolation score is bounded.

Feature: ueba-insider-threat-detection, Property 15: Daily risk equals the
configured weighted fusion.

Feature: ueba-insider-threat-detection, Property 16: Weekly risk equals the
trailing-7 daily sum.

Feature: ueba-insider-threat-detection, Property 17: Reason records are the true
top-3 contributors.

Feature: ueba-insider-threat-detection, Property 5: Artifact persistence
round-trips (risk.parquet).

These tests drive the Scoring_Engine (``src/ueba/scoring/engine.py``) directly.
Every example that needs an on-disk artifact writes into its own
``tempfile.TemporaryDirectory`` created inside the test body — the
function-scoped ``tmp_path`` fixture is deliberately avoided because combining
it with ``@given`` trips a Hypothesis health check.
"""

from __future__ import annotations

import dataclasses
import math
import os
import tempfile

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.ueba.baselines import Baseline
from src.ueba.config import FusionWeights, load_config
from src.ueba.features import FEATURE_COLUMNS
from src.ueba.models import train
from src.ueba.scoring import (
    RISK_FILENAME,
    compute_risk,
    daily_risk,
    reason_record,
    scale_isolation,
    weekly_risk,
    write_risk,
)
from src.ueba.scoring.engine import DEGENERATE_ISO_SCORE, RISK_COLUMNS, WEEKLY_WINDOW
from src.ueba.scoring.statistical import TOP_K, Z_CAP

# Repository root -> the project config.yaml, passed explicitly so these tests
# never depend on the process working directory (mirrors the other property
# suites in this package).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")

# Numeric tolerance for floating-point comparisons across the fusion arithmetic.
_TOL = 1e-9


# ---------------------------------------------------------------------------
# Property 14 — Isolation score is bounded (Req 6.3)
# ---------------------------------------------------------------------------

# Finite raw scores spanning negatives, zeros, and duplicates; sizes 0..50.
_raw_score_lists = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=50,
)


@settings(max_examples=200, deadline=None)
@given(raw=_raw_score_lists)
def test_isolation_score_is_bounded(raw):
    """Property 14: every scaled isolation score lies within [0, 1].

    Feature: ueba-insider-threat-detection, Property 14: Isolation score is
    bounded. Validates Requirements 6.3.

    For an arbitrary batch of raw IsolationForest scores, ``scale_isolation``
    maps every value into ``[0, 1]``. The degenerate all-equal batch (no spread)
    collapses to the neutral midpoint ``0.5``; an empty batch stays empty.
    """
    scaled = scale_isolation(raw)

    # One output per input, always.
    assert scaled.shape == (len(raw),)

    if len(raw) == 0:
        assert scaled.size == 0
        return

    # Bounded in [0, 1] (the core guarantee).
    assert np.all(scaled >= 0.0 - _TOL)
    assert np.all(scaled <= 1.0 + _TOL)

    lo, hi = min(raw), max(raw)
    if hi == lo:
        # No spread -> the whole batch is the neutral midpoint.
        assert np.allclose(scaled, DEGENERATE_ISO_SCORE)
    else:
        # The extreme days anchor the ends of the scaled range.
        assert math.isclose(scaled.min(), 0.0, abs_tol=_TOL)
        assert math.isclose(scaled.max(), 1.0, abs_tol=_TOL)


def test_isolation_score_edge_cases():
    """Property 14: explicit empty, single-element, and all-equal batches.

    Feature: ueba-insider-threat-detection, Property 14: Isolation score is
    bounded. Validates Requirements 6.3.
    """
    # Empty -> empty.
    assert scale_isolation([]).size == 0

    # Single element has no spread -> neutral midpoint.
    single = scale_isolation([7.3])
    assert single.shape == (1,)
    assert math.isclose(single[0], DEGENERATE_ISO_SCORE)

    # All-equal (incl. negatives) -> all 0.5, still bounded.
    allequal = scale_isolation([-4.0, -4.0, -4.0])
    assert np.allclose(allequal, DEGENERATE_ISO_SCORE)
    assert np.all((allequal >= 0.0) & (allequal <= 1.0))


# ---------------------------------------------------------------------------
# Property 15 — Daily risk equals the configured weighted fusion (Req 6.4/6.5)
# ---------------------------------------------------------------------------

_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def _fusion_weights(draw):
    """Draw a FusionWeights whose three weights lie in [0,1] and sum to 1.

    Normalising keeps the fused Daily_Risk inside ``[0, 100]`` (the design's
    bound), while still exercising arbitrary weight splits. An all-zero draw
    falls back to equal thirds.
    """
    raw = [draw(_unit), draw(_unit), draw(_unit)]
    total = sum(raw)
    if total == 0.0:
        raw = [1.0, 1.0, 1.0]
        total = 3.0
    self_w, iso_w, peer_w = (w / total for w in raw)
    return FusionWeights(self_w=self_w, iso_w=iso_w, peer_w=peer_w)


@settings(max_examples=200, deadline=None)
@given(self_s=_unit, iso_s=_unit, peer_s=_unit, weights=_fusion_weights())
def test_daily_risk_is_configured_weighted_fusion(self_s, iso_s, peer_s, weights):
    """Property 15 (scalar): daily_risk == 100*(w_self*s + w_iso*i + w_peer*p).

    Feature: ueba-insider-threat-detection, Property 15: Daily risk equals the
    configured weighted fusion. Validates Requirements 6.4, 6.5.

    The weights come from the passed FusionWeights (read from config in
    production, never hard-coded), and for in-range scores with weights summing
    to 1 the fused risk stays within ``[0, 100]``.
    """
    expected = 100.0 * (
        weights.self_w * self_s + weights.iso_w * iso_s + weights.peer_w * peer_s
    )
    got = daily_risk(self_s, iso_s, peer_s, weights)

    assert isinstance(got, float)
    assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9)
    assert -_TOL <= got <= 100.0 + _TOL


@settings(max_examples=100, deadline=None)
@given(
    selfs=st.lists(_unit, min_size=1, max_size=20),
    isos=st.lists(_unit, min_size=1, max_size=20),
    peers=st.lists(_unit, min_size=1, max_size=20),
    weights=_fusion_weights(),
)
def test_daily_risk_array_path(selfs, isos, peers, weights):
    """Property 15 (array): element-wise fusion over equal-length score arrays.

    Feature: ueba-insider-threat-detection, Property 15: Daily risk equals the
    configured weighted fusion. Validates Requirements 6.4, 6.5.
    """
    n = min(len(selfs), len(isos), len(peers))
    s = np.array(selfs[:n], dtype="float64")
    i = np.array(isos[:n], dtype="float64")
    p = np.array(peers[:n], dtype="float64")

    got = daily_risk(s, i, p, weights)
    expected = 100.0 * (weights.self_w * s + weights.iso_w * i + weights.peer_w * p)

    assert isinstance(got, np.ndarray)
    assert got.shape == (n,)
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-9)
    assert np.all(got >= -_TOL)
    assert np.all(got <= 100.0 + _TOL)


# ---------------------------------------------------------------------------
# Property 16 — Weekly risk equals the trailing-7 daily sum (Req 6.6)
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(
    daily=st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=40,
    )
)
def test_weekly_risk_is_trailing_seven_sum(daily):
    """Property 16: weekly_risk is the trailing-7-rows rolling sum of daily risk.

    Feature: ueba-insider-threat-detection, Property 16: Weekly risk equals the
    trailing-7 daily sum. Validates Requirements 6.6.

    Each element equals the sum of that value and up to six preceding values;
    the final element equals the sum of the user's last (up to) seven days.
    """
    series = pd.Series(daily, dtype="float64")
    weekly = weekly_risk(series)

    assert isinstance(weekly, pd.Series)
    assert len(weekly) == len(daily)

    # Independent trailing-7-rows rolling sum.
    expected = [
        sum(daily[max(0, idx - (WEEKLY_WINDOW - 1)) : idx + 1])
        for idx in range(len(daily))
    ]
    assert np.allclose(weekly.to_numpy(), np.array(expected), rtol=1e-9, atol=1e-6)

    # The final element is the sum of the last up-to-7 daily risks.
    assert math.isclose(
        weekly.iloc[-1], sum(daily[-WEEKLY_WINDOW:]), rel_tol=1e-9, abs_tol=1e-6
    )


# ---------------------------------------------------------------------------
# Property 17 — Reason records are the true top-3 contributors (Req 6.7)
# ---------------------------------------------------------------------------

# A std is either exactly zero (zero-variance feature -> z defined as 0) or a
# clearly positive spread; values and means span a moderate finite range.
_moderate = st.floats(
    min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
)
_std = st.one_of(
    st.just(0.0),
    st.floats(min_value=1e-3, max_value=1000.0, allow_nan=False, allow_infinity=False),
)


def _capped_z(value: float, mean: float, std: float) -> float:
    """Independent reimplementation of the scorer's capped z-score rule."""
    if std == 0.0:
        return 0.0
    z = (value - mean) / std
    if math.isnan(z):
        return 0.0
    return max(-Z_CAP, min(Z_CAP, z))


@st.composite
def _reason_cases(draw):
    """Draw (values, means, stds) dicts over every feature column."""
    values = {f: draw(_moderate) for f in FEATURE_COLUMNS}
    means = {f: draw(_moderate) for f in FEATURE_COLUMNS}
    stds = {f: draw(_std) for f in FEATURE_COLUMNS}
    return values, means, stds


@settings(max_examples=150, deadline=None)
@given(case=_reason_cases())
def test_reason_record_is_true_top3(case):
    """Property 17: the Reason_Record holds the true top-3 features by z-score.

    Feature: ueba-insider-threat-detection, Property 17: Reason records are the
    true top-3 contributors. Validates Requirements 6.7.

    For a user-day scored against a single-user self baseline, the reason record
    must list the (up to) three features with the highest capped z-scores, each
    carrying its correct value, baseline mean, and z-score.
    """
    values, means, stds = case
    user = "USER0001"

    baseline = Baseline(
        mean=pd.DataFrame([means], index=[user])[FEATURE_COLUMNS],
        std=pd.DataFrame([stds], index=[user])[FEATURE_COLUMNS],
        n_days=pd.Series({user: 42}),
        sufficient=None,
    )

    user_day = {"user": user, **values}

    reasons = reason_record(user_day, baseline)

    # Independent expectation: capped z per feature, sorted by descending z with
    # feature name as the deterministic tie-breaker (matching the scorer), top-3.
    scored = []
    for feature in FEATURE_COLUMNS:
        z = _capped_z(values[feature], means[feature], stds[feature])
        scored.append(
            {
                "feature": feature,
                "value": float(values[feature]),
                "baseline_mean": float(means[feature]),
                "z_score": z,
            }
        )
    scored.sort(key=lambda item: (-item["z_score"], item["feature"]))
    expected = scored[:TOP_K]

    assert len(reasons) <= TOP_K
    assert len(reasons) == len(expected)

    for got, exp in zip(reasons, expected):
        assert got["feature"] == exp["feature"]
        assert math.isclose(got["value"], exp["value"], rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(
            got["baseline_mean"], exp["baseline_mean"], rel_tol=1e-9, abs_tol=1e-9
        )
        assert math.isclose(
            got["z_score"], exp["z_score"], rel_tol=1e-9, abs_tol=1e-9
        )


# ---------------------------------------------------------------------------
# Property 5 — risk.parquet persistence round-trips (Req 6.8)
# ---------------------------------------------------------------------------


def _load_base_config():
    """Load the project config (falls back to repo-root config.yaml)."""
    return load_config(_CONFIG_PATH)


def _build_shared_model():
    """Train one small IsolationModel once, reused across Property 5 examples.

    Passing a pre-trained model into ``compute_risk`` avoids refitting a forest
    for every Hypothesis example. The model's actual scores are irrelevant to a
    round-trip test; only that ``compute_risk`` produces a well-formed frame.
    """
    rng = np.random.default_rng(0)
    rows = []
    for u in range(3):
        for d in range(20):
            row = {f: float(rng.integers(0, 10)) for f in FEATURE_COLUMNS}
            row["user"] = f"USER{u:04d}"
            rows.append(row)
    frame = pd.DataFrame(rows)
    return train(frame)


_SHARED_MODEL = _build_shared_model()
_BASE_CONFIG = _load_base_config()


def _normalize_reasons(reasons):
    """Return a list-of-plain-dicts view of one row's reasons for comparison."""
    out = []
    for entry in list(reasons):
        out.append(
            {
                "feature": str(entry["feature"]),
                "value": float(entry["value"]),
                "baseline_mean": float(entry["baseline_mean"]),
                "z_score": float(entry["z_score"]),
            }
        )
    return out


def _assert_risk_frames_equal(expected: pd.DataFrame, actual: pd.DataFrame):
    """Assert two risk frames match, including the list-of-struct reasons."""
    assert list(actual.columns) == RISK_COLUMNS
    assert len(actual) == len(expected)

    exp = expected.reset_index(drop=True)
    act = actual.reset_index(drop=True)

    assert [str(u) for u in act["user"]] == [str(u) for u in exp["user"]]
    assert list(pd.to_datetime(act["day"])) == list(pd.to_datetime(exp["day"]))

    for col in ("self_score", "isolation_score", "peer_score", "daily_risk", "weekly_risk"):
        assert np.allclose(
            act[col].to_numpy(dtype="float64"),
            exp[col].to_numpy(dtype="float64"),
            rtol=1e-9,
            atol=1e-9,
        ), f"column {col} did not round-trip"

    for idx in range(len(exp)):
        got = _normalize_reasons(act["reasons"].iloc[idx])
        want = _normalize_reasons(exp["reasons"].iloc[idx])
        assert len(got) == len(want), f"reason length mismatch at row {idx}"
        for g, w in zip(got, want):
            assert g["feature"] == w["feature"]
            assert math.isclose(g["value"], w["value"], rel_tol=1e-9, abs_tol=1e-9)
            assert math.isclose(
                g["baseline_mean"], w["baseline_mean"], rel_tol=1e-9, abs_tol=1e-9
            )
            assert math.isclose(
                g["z_score"], w["z_score"], rel_tol=1e-9, abs_tol=1e-9
            )


@st.composite
def _risk_frames(draw):
    """Draw a small user-day feature frame for compute_risk to fuse and persist."""
    n_users = draw(st.integers(min_value=1, max_value=3))
    n_days = draw(st.integers(min_value=2, max_value=8))
    start = pd.Timestamp("2020-01-01")

    rows = []
    for u in range(n_users):
        user = f"USER{u:04d}"
        role = f"ROLE{u % 2}"
        for d in range(n_days):
            row = {
                f: float(draw(st.integers(min_value=0, max_value=30)))
                for f in FEATURE_COLUMNS
            }
            row["user"] = user
            row["role"] = role
            row["day"] = start + pd.Timedelta(days=d)
            rows.append(row)

    columns = ["user", "day", "role", *FEATURE_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(features=_risk_frames())
def test_risk_parquet_round_trips(features):
    """Property 5: a computed risk frame round-trips through risk.parquet.

    Feature: ueba-insider-threat-detection, Property 5: Artifact persistence
    round-trips (risk.parquet). Validates Requirements 6.8.

    Fusing a feature table with ``compute_risk``, writing it with ``write_risk``,
    and reading it back must reproduce an equal frame — scores, daily/weekly
    risk, and the list-of-struct reasons alike.
    """
    computed = compute_risk(features, _BASE_CONFIG, model=_SHARED_MODEL)

    with tempfile.TemporaryDirectory() as processed_dir:
        config = dataclasses.replace(_BASE_CONFIG, processed_dir=processed_dir)
        out_path = write_risk(computed, config)
        assert os.path.basename(out_path) == RISK_FILENAME

        reloaded = pd.read_parquet(out_path)

    _assert_risk_frames_equal(computed, reloaded)


def test_risk_parquet_round_trips_empty_reasons():
    """Property 5: rows with an empty Reason_Record round-trip exactly.

    Feature: ueba-insider-threat-detection, Property 5: Artifact persistence
    round-trips (risk.parquet). Validates Requirements 6.8.

    The explicit list-of-struct schema must persist an empty reasons list (which
    would otherwise give the writer no struct shape to infer) alongside a
    populated one, and read both back unchanged.
    """
    frame = pd.DataFrame(
        {
            "user": pd.array(["USER0001", "USER0002"], dtype="string"),
            "day": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "self_score": [0.8, 0.0],
            "isolation_score": [0.6, 0.0],
            "peer_score": [0.4, 0.0],
            "daily_risk": [66.0, 0.0],
            "weekly_risk": [66.0, 0.0],
            "reasons": [
                [
                    {
                        "feature": "usb_connect_count",
                        "value": 30.0,
                        "baseline_mean": 1.0,
                        "z_score": 5.0,
                    }
                ],
                [],  # empty Reason_Record
            ],
        }
    )[RISK_COLUMNS]

    with tempfile.TemporaryDirectory() as processed_dir:
        config = dataclasses.replace(_BASE_CONFIG, processed_dir=processed_dir)
        out_path = write_risk(frame, config)
        reloaded = pd.read_parquet(out_path)

    _assert_risk_frames_equal(frame, reloaded)
    # The second row's reason list is genuinely empty after the round-trip.
    assert len(_normalize_reasons(reloaded["reasons"].iloc[1])) == 0
