"""Property-based and smoke tests for the Isolation_Model.

Feature: ueba-insider-threat-detection, Property 13: IsolationForest trains
only on baseline-period records.

Feature: ueba-insider-threat-detection, Property 18: Saved and reloaded model
score identically.

These tests drive the models layer (``src.ueba.models.isolation``) directly.
Each Hypothesis example builds an in-memory ``(user, day, role, features...)``
frame whose per-user histories deliberately span well beyond the baseline
window so that every frame contains both in-period and out-of-period rows. The
baseline split is cross-checked against ``baseline_cutoff`` (a different code
path from the ``select_training_rows`` under test), so the two must agree.

Every example that touches disk writes into its own
``tempfile.TemporaryDirectory`` created inside the test body — the
function-scoped ``tmp_path`` fixture is deliberately avoided because combining
it with ``@given`` trips a Hypothesis health check.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.ueba.baselines import baseline_cutoff
from src.ueba.config import load_config
from src.ueba.features import FEATURE_COLUMNS
from src.ueba.models import (
    default_model_path,
    load,
    save,
    score,
    select_training_rows,
    train_from_features,
)

# Repository root -> the project's config.yaml. Passed explicitly so the tests
# never depend on the process working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")

# Loaded once; Config is frozen and read-only so sharing it is safe.
_CONFIG = load_config(_CONFIG_PATH)

# All day offsets are relative to this fixed anchor date (CERT 4.2 start).
_BASE = pd.Timestamp("2010-01-04")

# A few arbitrary job roles so the frame carries the schema's role column.
_ROLES = ["ITAdmin", "Salesman", "Engineer"]

# The effective baseline window (target_days 75, clamped into 30-90) is 75 days
# inclusive, so a row is in-period when its offset from the user's first day is
# <= 74. Out-of-period deltas start at 75.
_LAST_IN_PERIOD_DELTA = 74
_FIRST_OUT_OF_PERIOD_DELTA = 75


@st.composite
def _user_day_frames(draw):
    """Draw an in-memory user-day feature frame spanning past the baseline.

    For a handful of users, each user gets its own first-active day and a mix of
    day offsets guaranteed to include at least one in-period day (delta 0, the
    anchor) and at least one out-of-period day (delta >= 75). Feature values are
    filled from a drawn seed via NumPy so the draw count stays small; their
    exact values are irrelevant to the baseline split and only need to be
    deterministic for the save/reload comparison.
    """
    n_users = draw(st.integers(min_value=1, max_value=3))
    seed = draw(st.integers(min_value=0, max_value=10_000))

    skeleton: list[tuple[str, pd.Timestamp, str]] = []
    for u in range(n_users):
        user = f"USER{u:04d}"
        role = draw(st.sampled_from(_ROLES))
        start_offset = draw(st.integers(min_value=0, max_value=20))

        in_deltas = draw(
            st.sets(
                st.integers(min_value=0, max_value=_LAST_IN_PERIOD_DELTA),
                min_size=1,
                max_size=6,
            )
        )
        in_deltas.add(0)  # the anchor day is always in-period
        out_deltas = draw(
            st.sets(
                st.integers(min_value=_FIRST_OUT_OF_PERIOD_DELTA, max_value=160),
                min_size=1,
                max_size=4,
            )
        )

        for delta in sorted(in_deltas | out_deltas):
            day = _BASE + pd.Timedelta(days=int(start_offset + delta))
            skeleton.append((user, day, role))

    rng = np.random.default_rng(seed)
    rows = []
    for user, day, role in skeleton:
        feats = {col: int(rng.integers(0, 50)) for col in FEATURE_COLUMNS}
        rows.append({"user": user, "day": day, "role": role, **feats})

    return pd.DataFrame(rows, columns=["user", "day", "role", *FEATURE_COLUMNS])


def _cutoffs_by_user(frame: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Independently computed inclusive baseline cutoff for every user."""
    return {
        user: baseline_cutoff(group, _CONFIG)
        for user, group in frame.groupby("user", sort=False)
    }


def _user_day_keys(frame: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    """Normalise a frame to the set of its (user, day) identity pairs."""
    return {
        (row["user"], pd.Timestamp(row["day"]))
        for _, row in frame.iterrows()
    }


# ---------------------------------------------------------------------------
# Property 13
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(frame=_user_day_frames())
def test_isolation_trains_only_on_baseline_rows(frame):
    """Property 13: training rows are exactly the baseline-period rows.

    Feature: ueba-insider-threat-detection, Property 13: IsolationForest trains
    only on baseline-period records. Validates Requirements 6.1, 6.2.

    For any set of dated user-days, ``select_training_rows`` must return exactly
    the records dated on or before each user's inclusive Baseline_Cutoff and
    exclude every record dated after it.
    """
    cutoffs = _cutoffs_by_user(frame)

    # Independently computed oracle: in-period rows per user via baseline_cutoff.
    expected = set()
    for user, group in frame.groupby("user", sort=False):
        cutoff = cutoffs[user]
        in_period = group[pd.to_datetime(group["day"]) <= cutoff]
        expected |= _user_day_keys(in_period)

    training = select_training_rows(frame, _CONFIG)
    actual = _user_day_keys(training)

    # Exact match: no post-cutoff row leaked in, and no in-period row went missing.
    assert actual == expected

    # And defensively: every returned row sits on or before its user's cutoff.
    for _, row in training.iterrows():
        day = pd.Timestamp(row["day"])
        assert day <= cutoffs[row["user"]], (
            f"training row for {row['user']} dated {day} is after its cutoff "
            f"{cutoffs[row['user']]}"
        )

    # The frames are built to always include out-of-period rows, so the split
    # must actually drop something (guards against a vacuous pass).
    assert len(training) < len(frame)


# ---------------------------------------------------------------------------
# Property 18
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(frame=_user_day_frames())
def test_saved_and_reloaded_model_score_identically(frame):
    """Property 18: a reloaded model scores identically to the in-memory one.

    Feature: ueba-insider-threat-detection, Property 18: Saved and reloaded
    model score identically. Validates Requirements 6.10.

    For any set of user-days, scoring with a model reloaded from disk must
    produce exactly the same scores as the in-memory model it was saved from.
    """
    # n_estimators is reduced purely for speed; determinism comes from the
    # bundle's fixed random_state and the self-contained scaler + columns.
    in_memory = train_from_features(frame, _CONFIG, n_estimators=50)
    scores_memory = score(in_memory, frame)

    with tempfile.TemporaryDirectory() as model_dir:
        path = Path(model_dir) / "isolation_model.joblib"
        saved_path = save(in_memory, path)
        assert saved_path.is_file()
        reloaded = load(saved_path)

    scores_reloaded = score(reloaded, frame)

    assert np.array_equal(scores_memory, scores_reloaded), (
        "reloaded model produced different scores than the in-memory model"
    )


# ---------------------------------------------------------------------------
# Smoke test: model file saved (Req 6.9)
# ---------------------------------------------------------------------------


def _small_training_frame() -> pd.DataFrame:
    """A tiny, fully in-period single-user frame suitable for training."""
    rows = []
    for delta in range(35):  # 35 consecutive in-period days (<= 74)
        day = _BASE + pd.Timedelta(days=delta)
        feats = {col: (delta + i) % 7 for i, col in enumerate(FEATURE_COLUMNS)}
        rows.append({"user": "USER0000", "day": day, "role": "Engineer", **feats})
    return pd.DataFrame(rows, columns=["user", "day", "role", *FEATURE_COLUMNS])


def test_model_file_saved_to_disk_and_loadable():
    """Smoke: training then saving writes the model file and it can be loaded.

    Feature: ueba-insider-threat-detection, Smoke (model saved). Validates
    Requirements 6.9.

    Training a model and saving it to ``<processed_dir>/isolation_model.joblib``
    must write a real file on disk at the default location, and loading it back
    must yield a model that scores. ``processed_dir`` is redirected to a temp
    directory so nothing is written into the repository.
    """
    frame = _small_training_frame()

    with tempfile.TemporaryDirectory() as processed_dir:
        config = dataclasses.replace(_CONFIG, processed_dir=processed_dir)

        model = train_from_features(frame, config, n_estimators=50)
        expected_path = default_model_path(config)
        saved_path = save(model, expected_path)

        assert saved_path == expected_path
        assert saved_path.is_file()
        assert saved_path.name == "isolation_model.joblib"

        reloaded = load(saved_path)

    scores = score(reloaded, frame)
    assert scores.shape[0] == len(frame)
    assert np.all(np.isfinite(scores))
