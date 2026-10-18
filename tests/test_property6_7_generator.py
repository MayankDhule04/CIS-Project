"""Property-based tests for the Synthetic_Generator.

Feature: ueba-insider-threat-detection, Property 6: Synthetic logs match the
dataset schema.

Feature: ueba-insider-threat-detection, Property 7: Planted anomalies create
measurable deviation.

These tests drive ``scripts/generate_synthetic.py`` directly through the
``SyntheticGenerator`` class (never the module-level singleton) so each
Hypothesis example gets an isolated generator with no shared state. Every
example writes into its own ``tempfile.TemporaryDirectory`` created inside the
test body — the function-scoped ``tmp_path`` fixture is deliberately avoided
because combining it with ``@given`` trips a Hypothesis health check.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timedelta

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from generate_synthetic import SyntheticGenerator

# Repository root -> the config.yaml the generator reads. Passed explicitly so
# the test does not depend on the process working directory. (The generator
# falls back to built-in defaults if the file is unreadable, so the schema and
# the hard-coded USB burst size are stable either way.)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")

# The generator's CERT release 4.2 start date (see _DEFAULT_START_DATE).
_START_DATE = date(2010, 1, 4)

# CERT-style timestamp written into the event CSVs.
_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

# The number of Connect events a usb_burst plants on the target user-day.
_USB_BURST_CONNECTS = 30

# Declared CERT 4.2 per-file schema, derived independently of the module's own
# column constants so the test genuinely validates the produced files.
_EXPECTED_SCHEMAS = {
    "logon.csv": ["id", "date", "user", "pc", "activity"],
    "device.csv": ["id", "date", "user", "pc", "activity"],
    "http.csv": ["id", "date", "user", "pc", "url"],
    "email.csv": [
        "id",
        "date",
        "user",
        "pc",
        "to",
        "cc",
        "bcc",
        "from",
        "size",
        "attachments",
        "content",
    ],
    "roles.csv": ["user", "role"],
    "labels.csv": ["user", "day", "insider"],
}


def _user_name(index: int) -> str:
    """Generator user id for a 1-based index, e.g. 5 -> 'USER0005'."""
    return f"USER{index:04d}"


# ---------------------------------------------------------------------------
# Property 6
# ---------------------------------------------------------------------------


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    n_users=st.integers(min_value=1, max_value=5),
    n_days=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_synthetic_logs_match_schema(n_users, n_days, seed):
    """Property 6: every produced file's columns match the declared schema.

    Feature: ueba-insider-threat-detection, Property 6: Synthetic logs match
    the dataset schema. Validates Requirements 3.1.

    For arbitrary generation parameters (n_users, n_days, seed), each CSV the
    generator writes must carry exactly the declared CERT 4.2 column order for
    that file type.
    """
    with tempfile.TemporaryDirectory() as out_dir:
        gen = SyntheticGenerator(config_path=_CONFIG_PATH)
        gen.generate(out_dir, n_users=n_users, n_days=n_days, seed=seed)

        for filename, expected_columns in _EXPECTED_SCHEMAS.items():
            path = os.path.join(out_dir, filename)
            assert os.path.exists(path), f"missing output file: {filename}"

            # Read only the header row; column order and names are the subject.
            columns = pd.read_csv(path, nrows=0).columns.tolist()
            assert columns == expected_columns, (
                f"{filename} columns {columns} != expected {expected_columns}"
            )


# ---------------------------------------------------------------------------
# Property 7
# ---------------------------------------------------------------------------


@st.composite
def _usb_burst_cases(draw):
    """Draw (n_users, n_days, seed, user_id, day_offset) for a USB-burst plant.

    ``n_days`` stays in the 90-150 range so the target user has a substantial
    baseline of non-planted days to compare against.
    """
    n_users = draw(st.integers(min_value=1, max_value=3))
    n_days = draw(st.integers(min_value=90, max_value=150))
    seed = draw(st.integers(min_value=0, max_value=10_000))
    user_index = draw(st.integers(min_value=1, max_value=n_users))
    day_offset = draw(st.integers(min_value=0, max_value=n_days - 1))
    return n_users, n_days, seed, _user_name(user_index), day_offset


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(case=_usb_burst_cases())
def test_planted_anomaly_exceeds_baseline(case):
    """Property 7: a planted USB burst far exceeds the user's own baseline.

    Feature: ueba-insider-threat-detection, Property 7: Planted anomalies
    create measurable deviation. Validates Requirements 3.2, 5.6.

    After planting a ``usb_burst`` for a user on a day, that user-day's USB
    Connect count must exceed the user's baseline distribution for that
    feature — specifically it must sit above mean + 4*std of the user's other
    days and above every other day's count.
    """
    n_users, n_days, seed, user_id, day_offset = case
    planted_day = _START_DATE + timedelta(days=day_offset)

    with tempfile.TemporaryDirectory() as out_dir:
        gen = SyntheticGenerator(config_path=_CONFIG_PATH)
        gen.generate(out_dir, n_users=n_users, n_days=n_days, seed=seed)
        gen.plant_anomaly(user_id, day_offset, "usb_burst")

        device = pd.read_csv(os.path.join(out_dir, "device.csv"))

    # Connect events for the target user only.
    user_connects = device[
        (device["user"] == user_id) & (device["activity"] == "Connect")
    ].copy()

    # Map each event to its calendar day, then count Connects per day over the
    # whole generated span (days with no USB use count as 0, which only
    # tightens the baseline and makes the test stricter).
    user_connects["day"] = pd.to_datetime(
        user_connects["date"], format=_DATE_FORMAT
    ).dt.date
    per_day_counts = user_connects.groupby("day").size()

    all_days = [_START_DATE + timedelta(days=d) for d in range(n_days)]
    per_day = per_day_counts.reindex(all_days, fill_value=0)

    planted_count = int(per_day.loc[planted_day])
    other_days = per_day.drop(planted_day)

    baseline_mean = float(other_days.mean())
    baseline_std = float(other_days.std(ddof=0))
    max_other = int(other_days.max()) if len(other_days) else 0

    # The burst plants a fixed number of Connects on top of any normal activity.
    assert planted_count >= _USB_BURST_CONNECTS, (
        f"planted day had {planted_count} connects, "
        f"expected at least {_USB_BURST_CONNECTS}"
    )
    # Far above the baseline distribution (z > 4) and above every other day.
    assert planted_count > baseline_mean + 4 * baseline_std, (
        f"planted count {planted_count} did not exceed baseline "
        f"mean+4*std = {baseline_mean + 4 * baseline_std:.3f}"
    )
    assert planted_count > max_other, (
        f"planted count {planted_count} did not exceed the max of other "
        f"days ({max_other})"
    )
