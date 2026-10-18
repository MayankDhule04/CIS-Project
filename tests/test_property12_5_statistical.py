"""Property-based tests for the Statistical_Scorer.

Feature: ueba-insider-threat-detection, Property 12: Statistical score
invariants hold for self and peer.

Feature: ueba-insider-threat-detection, Property 5: Artifact persistence
round-trips (``scores_stat.parquet``).

These tests drive ``src/ueba/scoring/statistical.py`` directly. Property 12
builds ``Baseline`` objects by hand (so zero-variance and cap cases can be
forced) and recomputes the expected capped z-scores, top-3 selection, and
scaled score independently, asserting the scorer agrees and that every returned
z sits in ``[-5, 5]`` and every score in ``[0, 1]``. Property 5 scores a small
in-memory feature frame, writes it to ``scores_stat.parquet`` through
``write_scores_stat`` (into a temp ``processed_dir``), reads it back, and asserts
the frame round-trips exactly — including the list-of-struct top-feature columns
and rows whose top-feature list is empty.

Every example that touches disk creates its own ``tempfile.TemporaryDirectory``
inside the test body; the function-scoped ``tmp_path`` fixture is deliberately
avoided because combining it with ``@given`` trips a Hypothesis health check.
"""

from __future__ import annotations

import dataclasses
import math
import os
import tempfile

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ueba.baselines import Baseline, build_peer_baseline, build_self_baseline
from ueba.config import load_config
from ueba.features import FEATURE_COLUMNS
from ueba.scoring.statistical import (
    Z_CAP,
    peer_score,
    score_statistical,
    self_score,
    write_scores_stat,
)

# Repository root -> the project config.yaml, passed explicitly so the test
# does not depend on the process working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")

# Top-K features that contribute to a score (mirrors statistical.TOP_K).
_TOP_K = 3

# Float comparison tolerance for recomputed scores / z-scores.
_ABS_TOL = 1e-9
_REL_TOL = 1e-9


# ---------------------------------------------------------------------------
# Independent reference implementation of the scoring rule (no shared code
# with the module under test, so the test genuinely checks the invariants).
# ---------------------------------------------------------------------------


def _ref_capped_z(value: float, mean: float, std: float) -> float:
    """Reference capped z-score: std==0 -> 0, else (v-m)/s clamped to [-5, 5]."""
    if std == 0.0 or not math.isfinite(std):
        return 0.0
    z = (value - mean) / std
    if not math.isfinite(z):
        return 0.0
    return max(-Z_CAP, min(Z_CAP, z))


def _ref_score_and_top(values, means, stds):
    """Recompute (score, top_features) over FEATURE_COLUMNS independently."""
    scored = []
    for feature, value, mean, std in zip(FEATURE_COLUMNS, values, means, stds):
        scored.append(
            {
                "feature": feature,
                "value": float(value),
                "baseline_mean": float(mean),
                "z_score": _ref_capped_z(float(value), float(mean), float(std)),
            }
        )
    # Most-positive deviation first; ties broken by feature name (ascending).
    scored.sort(key=lambda item: (-item["z_score"], item["feature"]))
    top = scored[:_TOP_K]
    mean_top_z = sum(item["z_score"] for item in top) / len(top)
    score = max(0.0, min(1.0, mean_top_z / Z_CAP))
    return score, top


def _approx(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)


def _assert_top_equal(actual, expected) -> None:
    assert len(actual) == len(expected), (
        f"top-feature count {len(actual)} != expected {len(expected)}"
    )
    for got, exp in zip(actual, expected):
        assert got["feature"] == exp["feature"], (
            f"feature {got['feature']} != {exp['feature']}"
        )
        assert _approx(got["value"], exp["value"])
        assert _approx(got["baseline_mean"], exp["baseline_mean"])
        assert _approx(got["z_score"], exp["z_score"])


# ---------------------------------------------------------------------------
# Property 12 — statistical score invariants
# ---------------------------------------------------------------------------


@st.composite
def _baseline_cases(draw):
    """Draw (values, means, stds) each of length ``len(FEATURE_COLUMNS)``.

    ``std`` is drawn so a meaningful fraction of features are exactly zero,
    forcing the zero-variance rule, while values and means range widely enough
    (relative to small stds) to push some z-scores past the +/-5 cap.
    """
    n = len(FEATURE_COLUMNS)
    finite = dict(allow_nan=False, allow_infinity=False)
    values = draw(
        st.lists(
            st.floats(min_value=-1000.0, max_value=1000.0, **finite),
            min_size=n,
            max_size=n,
        )
    )
    means = draw(
        st.lists(
            st.floats(min_value=-1000.0, max_value=1000.0, **finite),
            min_size=n,
            max_size=n,
        )
    )
    stds = draw(
        st.lists(
            st.one_of(
                st.just(0.0),
                st.floats(min_value=0.01, max_value=500.0, **finite),
            ),
            min_size=n,
            max_size=n,
        )
    )
    return values, means, stds


def _make_baseline(group_key: str, means, stds) -> Baseline:
    """A one-group Baseline indexed by ``group_key`` over FEATURE_COLUMNS."""
    mean_df = pd.DataFrame([list(means)], index=[group_key], columns=FEATURE_COLUMNS)
    std_df = pd.DataFrame([list(stds)], index=[group_key], columns=FEATURE_COLUMNS)
    n_days = pd.Series({group_key: 1}, name="n_days")
    return Baseline(mean=mean_df, std=std_df, n_days=n_days)


@settings(max_examples=150, deadline=None)
@given(case=_baseline_cases())
def test_statistical_score_invariants(case):
    """Property 12: capped z-scores and top-3 scaling hold for self and peer.

    Feature: ueba-insider-threat-detection, Property 12: Statistical score
    invariants hold for self and peer. Validates Requirements 5.3, 5.4.

    For any user-day and baseline (own or role-group), each feature z-score is
    capped at 5 and the score equals the mean of the top-3 capped z-scores
    scaled into ``[0, 1]`` (mean(top-3 capped z) / 5). The same baseline, keyed
    by a shared group id, is scored through both ``self_score`` (keyed on
    ``user``) and ``peer_score`` (keyed on ``role``).
    """
    values, means, stds = case
    group_key = "G"
    baseline = _make_baseline(group_key, means, stds)

    user_day = {"user": group_key, "role": group_key}
    for feature, value in zip(FEATURE_COLUMNS, values):
        user_day[feature] = value

    expected_score, expected_top = _ref_score_and_top(values, means, stds)

    for score_fn in (self_score, peer_score):
        score, top = score_fn(user_day, baseline)

        # Score lands in [0, 1] and matches the independent recomputation.
        assert 0.0 <= score <= 1.0, f"score {score} outside [0, 1]"
        assert _approx(score, expected_score), (
            f"{score_fn.__name__} score {score} != expected {expected_score}"
        )

        # Every returned z-score is capped into [-5, 5].
        for item in top:
            assert -Z_CAP <= item["z_score"] <= Z_CAP, (
                f"z-score {item['z_score']} outside [-{Z_CAP}, {Z_CAP}]"
            )

        # Top-3 selection (and each field) matches the reference.
        _assert_top_equal(top, expected_top)


# ---------------------------------------------------------------------------
# Property 5 — scores_stat.parquet round-trips
# ---------------------------------------------------------------------------


@st.composite
def _feature_frames(draw):
    """Draw a small user-day feature frame (users, roles, days, features).

    Every user gets distinct calendar days and a role from a small pool, so the
    self/peer baselines built from the frame are well defined.
    """
    n_users = draw(st.integers(min_value=1, max_value=3))
    roles_pool = ["Analyst", "Engineer", "Manager"]

    records: list[dict] = []
    for u in range(n_users):
        user = f"USER{u:03d}"
        role = draw(st.sampled_from(roles_pool))
        n_days = draw(st.integers(min_value=1, max_value=4))
        for d in range(n_days):
            row = {
                "user": user,
                "day": pd.Timestamp("2010-01-01") + pd.Timedelta(days=d),
                "role": role,
            }
            for feature in FEATURE_COLUMNS:
                row[feature] = float(draw(st.integers(min_value=0, max_value=100)))
            records.append(row)

    return pd.DataFrame(records)


def _normalize_structs(cell) -> list[dict]:
    """Normalise a list-of-struct cell to a plain list of float-valued dicts.

    Handles ``None``/NaN, pandas/numpy array cells, and lists of dicts so the
    written and read-back values can be compared element-wise.
    """
    if cell is None:
        return []
    if isinstance(cell, float) and math.isnan(cell):
        return []
    out = []
    for entry in list(cell):
        out.append(
            {
                "feature": str(entry["feature"]),
                "value": float(entry["value"]),
                "baseline_mean": float(entry["baseline_mean"]),
                "z_score": float(entry["z_score"]),
            }
        )
    return out


def _assert_scores_equal(written: pd.DataFrame, read_back: pd.DataFrame) -> None:
    assert len(written) == len(read_back), "row count changed on round-trip"
    w = written.reset_index(drop=True)
    r = read_back.reset_index(drop=True)

    for i in range(len(w)):
        assert str(w.loc[i, "user"]) == str(r.loc[i, "user"])
        assert pd.Timestamp(w.loc[i, "day"]) == pd.Timestamp(r.loc[i, "day"])

        for col in ("self_score", "peer_score"):
            wv = float(w.loc[i, col])
            rv = float(r.loc[i, col])
            if math.isnan(wv) or math.isnan(rv):
                assert math.isnan(wv) and math.isnan(rv), (
                    f"row {i} {col}: {wv} vs {rv}"
                )
            else:
                assert _approx(wv, rv), f"row {i} {col}: {wv} vs {rv}"

        for col in ("self_top_features", "peer_top_features"):
            wt = _normalize_structs(w.loc[i, col])
            rt = _normalize_structs(r.loc[i, col])
            _assert_top_equal(rt, wt)


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(frame=_feature_frames())
def test_scores_stat_roundtrips(frame):
    """Property 5: scores written to scores_stat.parquet read back unchanged.

    Feature: ueba-insider-threat-detection, Property 5: Artifact persistence
    round-trips (scores_stat.parquet). Validates Requirements 5.5.

    Baselines are built from the generated frame; a "stranger" user-day whose
    user and role are absent from the baselines is appended so the frame also
    carries NaN scores with empty top-feature lists. The scored frame is written
    through ``write_scores_stat`` into a temp ``processed_dir`` and read back;
    the two frames must be equal, including the list-of-struct columns.
    """
    self_baseline = build_self_baseline(frame)
    peer_baseline = build_peer_baseline(frame)

    # A user-day absent from both baselines -> NaN scores and empty top lists,
    # exercising the empty list<struct> round-trip path.
    stranger = {
        "user": "STRANGER",
        "day": pd.Timestamp("2010-02-01"),
        "role": "ROLE_UNKNOWN",
    }
    for feature in FEATURE_COLUMNS:
        stranger[feature] = 7.0
    scoring_frame = pd.concat(
        [frame, pd.DataFrame([stranger])], ignore_index=True
    )

    scores = score_statistical(scoring_frame, self_baseline, peer_baseline)

    # Sanity: the stranger row must have landed as an empty-top-features row.
    stranger_rows = scores[scores["user"] == "STRANGER"]
    assert len(stranger_rows) == 1
    assert list(stranger_rows.iloc[0]["self_top_features"]) == []
    assert list(stranger_rows.iloc[0]["peer_top_features"]) == []

    base_config = load_config(_CONFIG_PATH)
    with tempfile.TemporaryDirectory() as processed_dir:
        config = dataclasses.replace(base_config, processed_dir=processed_dir)
        out_path = write_scores_stat(scores, config)
        assert os.path.exists(out_path)

        read_back = pd.read_parquet(out_path)

    _assert_scores_equal(scores, read_back)
