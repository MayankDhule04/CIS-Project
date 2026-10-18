"""Property-based tests for the Ingestion_Pipeline and subset selection.

Feature: ueba-insider-threat-detection, Property 2: Aggregation yields exactly
one record per user-day.

Feature: ueba-insider-threat-detection, Property 3: Null-user events are
excluded without side effects.

Feature: ueba-insider-threat-detection, Property 4: Subset includes every
insider.

Properties 2 and 3 drive ``ueba.ingestion.pipeline.ingest`` directly. Rather
than leaning on the Synthetic_Generator, each Hypothesis example builds a tiny
set of raw event CSVs by hand so the exact ``(user, day)`` pairs — and the
injected null/blank users — are fully under the test's control. Every example
writes into its own ``tempfile.TemporaryDirectory`` created inside the test
body (the function-scoped ``tmp_path`` fixture is deliberately avoided because
combining it with ``@given`` trips a Hypothesis health check), and points a
``dataclasses.replace`` copy of the real ``Config`` at those temp dirs.

Property 4 exercises ``ueba.ingestion.subset.select_subset`` directly.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from collections import defaultdict
from datetime import date, datetime, time, timedelta

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ueba.config import load_config
from ueba.ingestion.pipeline import ingest
from ueba.ingestion.subset import select_subset

# Repository root -> the real config.yaml, which supplies the column_map the
# pipeline uses to find the user/date columns. We only override the raw and
# processed directories per example via dataclasses.replace (Config is frozen).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")
_BASE_CONFIG = load_config(_CONFIG_PATH)

# CERT release 4.2 timestamp layout the pipeline parses first.
_DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
_START_DATE = date(2010, 1, 4)
_EVENT_HOUR = time(10, 0, 0)

# Event file -> the per-day count column it feeds in user_day.parquet, in the
# same stable order the pipeline uses.
_COUNT_COLS = {
    "logon": "logon_events",
    "device": "device_events",
    "http": "http_events",
    "email": "email_events",
}
_COUNT_INDEX = {ftype: i for i, ftype in enumerate(_COUNT_COLS)}

# A small pool of valid user ids plus blank/whitespace markers that must be
# treated as missing. ``None`` stands in for a truly absent id.
_VALID_USERS = ["U0", "U1", "U2", "U3"]
_BLANK_USERS = ["", "   ", None]


# ---------------------------------------------------------------------------
# Raw-CSV construction helpers
# ---------------------------------------------------------------------------


def _timestamp_for(day_offset: int) -> str:
    """CERT-format timestamp string for a day offset from the start date."""
    stamp = datetime.combine(_START_DATE + timedelta(days=day_offset), _EVENT_HOUR)
    return stamp.strftime(_DATE_FORMAT)


def _rows_for_file(file_type: str, rows: list[tuple[object, int]]) -> pd.DataFrame:
    """Build a raw-CSV DataFrame for one event file in its CERT 4.2 schema.

    ``rows`` is a list of ``(user, day_offset)`` pairs; ``user`` may be a valid
    id string, an empty/whitespace string, or ``None`` (written as a blank
    field). Only the user and date columns matter to ingestion, but the full
    declared schema is written so the fixtures look like real data.
    """
    n = len(rows)
    users = [r[0] for r in rows]
    dates = [_timestamp_for(r[1]) for r in rows]
    ids = [f"{file_type[:1].upper()}{i}" for i in range(n)]

    if file_type in ("logon", "device"):
        return pd.DataFrame(
            {
                "id": ids,
                "date": dates,
                "user": users,
                "pc": ["PC-1"] * n,
                "activity": ["Logon"] * n,
            }
        )
    if file_type == "http":
        return pd.DataFrame(
            {
                "id": ids,
                "date": dates,
                "user": users,
                "pc": ["PC-1"] * n,
                "url": ["http://example.com"] * n,
            }
        )
    # email: the widest schema.
    return pd.DataFrame(
        {
            "id": ids,
            "date": dates,
            "user": users,
            "pc": ["PC-1"] * n,
            "to": ["a@x.com"] * n,
            "cc": [""] * n,
            "bcc": [""] * n,
            "from": ["b@x.com"] * n,
            "size": [100] * n,
            "attachments": [0] * n,
            "content": ["body"] * n,
        }
    )


def _write_event_csvs(raw_dir: str, events: list[tuple[str, object, int]]) -> None:
    """Write grouped ``events`` as the four raw event CSVs under ``raw_dir``.

    ``events`` is a flat list of ``(file_type, user, day_offset)`` triples. A
    file is written only when it has at least one event, mirroring how a real
    raw directory may be missing a source entirely.
    """
    grouped: dict[str, list[tuple[object, int]]] = defaultdict(list)
    for file_type, user, day_offset in events:
        grouped[file_type].append((user, day_offset))

    for file_type, rows in grouped.items():
        frame = _rows_for_file(file_type, rows)
        frame.to_csv(os.path.join(raw_dir, f"{file_type}.csv"), index=False)


def _config_for(raw_dir: str, processed_dir: str):
    """A frozen Config copy pointed at the given temp directories."""
    return dataclasses.replace(
        _BASE_CONFIG, raw_data_dir=raw_dir, processed_dir=processed_dir
    )


def _result_map(df: pd.DataFrame) -> dict[tuple[str, date], tuple[int, ...]]:
    """Map a user_day frame to ``{(user, day): (logon, device, http, email, total)}``."""
    result: dict[tuple[str, date], tuple[int, ...]] = {}
    for row in df.itertuples(index=False):
        key = (str(row.user), pd.Timestamp(row.day).date())
        result[key] = (
            int(row.logon_events),
            int(row.device_events),
            int(row.http_events),
            int(row.email_events),
            int(row.total_events),
        )
    return result


def _is_missing(user: object) -> bool:
    """True when a raw user id is null/blank and should be dropped."""
    return user is None or str(user).strip() == ""


def _truth_map(events: list[tuple[str, object, int]]) -> dict[tuple[str, date], tuple[int, ...]]:
    """Independently compute the expected user_day map from raw ``events``."""
    counts: dict[tuple[str, date], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for file_type, user, day_offset in events:
        if _is_missing(user):
            continue
        key = (str(user).strip(), _START_DATE + timedelta(days=day_offset))
        counts[key][_COUNT_INDEX[file_type]] += 1
    return {k: (*c, sum(c)) for k, c in counts.items()}


# ---------------------------------------------------------------------------
# Event strategies
# ---------------------------------------------------------------------------

_event = st.tuples(
    st.sampled_from(list(_COUNT_COLS)),
    st.sampled_from(_VALID_USERS + _BLANK_USERS),
    st.integers(min_value=0, max_value=9),
)
_events = st.lists(_event, min_size=0, max_size=40)


# ---------------------------------------------------------------------------
# Property 2
# ---------------------------------------------------------------------------


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(events=_events)
def test_aggregation_one_record_per_user_day(events):
    """Property 2: ingest yields exactly one row per distinct user-day.

    Feature: ueba-insider-threat-detection, Property 2: Aggregation yields
    exactly one record per user-day. Validates Requirements 2.2, 2.4.

    For an arbitrary set of raw events, ``user_day.parquet`` must contain
    exactly one record for each distinct non-null ``(user, calendar_day)`` pair
    present in the input and no others, with no duplicate pairs.
    """
    with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as proc_dir:
        _write_event_csvs(raw_dir, events)
        result = ingest(_config_for(raw_dir, proc_dir))

    truth = _truth_map(events)

    # No duplicate (user, day) pairs: one physical row per logical pair.
    pairs = list(zip(result["user"].astype(str), result["day"]))
    assert len(pairs) == len(set(pairs)), "duplicate (user, day) rows found"
    assert len(result) == len(truth), (
        f"row count {len(result)} != distinct user-day pairs {len(truth)}"
    )

    # Exactly the expected pairs, with exactly the expected per-source counts.
    assert _result_map(result) == truth


# ---------------------------------------------------------------------------
# Property 3
# ---------------------------------------------------------------------------


@st.composite
def _events_with_nulls(draw):
    """Draw raw events guaranteed to contain at least one null/blank user.

    Mixing in a guaranteed null row makes every example genuinely exercise the
    exclusion path rather than occasionally reducing to the no-null case.
    """
    events = draw(_events)
    null_user = draw(st.sampled_from(_BLANK_USERS))
    file_type = draw(st.sampled_from(list(_COUNT_COLS)))
    day_offset = draw(st.integers(min_value=0, max_value=9))
    events = list(events) + [(file_type, null_user, day_offset)]
    draw(st.randoms(use_true_random=True)).shuffle(events)
    return events


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(events=_events_with_nulls())
def test_null_users_excluded_without_side_effects(events):
    """Property 3: null-user rows vanish and leave surviving counts untouched.

    Feature: ueba-insider-threat-detection, Property 3: Null-user events are
    excluded without side effects. Validates Requirements 2.5.

    Records with a null/blank user id must not appear in the user_day output,
    and removing them from the input must not change any surviving user-day's
    counts. We verify this by ingesting the full events and, separately, the
    same events with null/blank users stripped, then comparing the two outputs.
    """
    events_without_nulls = [e for e in events if not _is_missing(e[1])]

    with tempfile.TemporaryDirectory() as raw_a, tempfile.TemporaryDirectory() as proc_a:
        _write_event_csvs(raw_a, events)
        with_nulls = ingest(_config_for(raw_a, proc_a))

    with tempfile.TemporaryDirectory() as raw_b, tempfile.TemporaryDirectory() as proc_b:
        _write_event_csvs(raw_b, events_without_nulls)
        without_nulls = ingest(_config_for(raw_b, proc_b))

    # No surviving row carries a null/blank user id.
    assert with_nulls["user"].notna().all()
    assert (with_nulls["user"].astype(str).str.strip() != "").all()

    # Dropping the null-user rows is a no-op on every surviving user-day.
    assert _result_map(with_nulls) == _result_map(without_nulls)

    # And both agree with the independently computed non-null truth.
    assert _result_map(with_nulls) == _truth_map(events_without_nulls)


# ---------------------------------------------------------------------------
# Property 4
# ---------------------------------------------------------------------------

_user_ids = st.builds(lambda n: f"U{n}", st.integers(min_value=0, max_value=60))


@settings(max_examples=200, deadline=None)
@given(
    population=st.lists(_user_ids, max_size=40),
    insiders=st.lists(_user_ids, max_size=20),
    sample_size=st.integers(min_value=-5, max_value=50),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_subset_includes_every_insider(population, insiders, sample_size, seed):
    """Property 4: select_subset keeps every insider and caps sampled normals.

    Feature: ueba-insider-threat-detection, Property 4: Subset includes every
    insider. Validates Requirements 2.6.

    For any user population and insider set, ``select_subset`` includes every
    insider and adds the configured number of sampled normals, capped at
    ``sample_size`` (or all normals when fewer exist). Selection is also
    deterministic under a fixed seed.
    """
    pop_set = set(population)
    insider_set = set(insiders)

    result = select_subset(pop_set, insider_set, sample_size, seed)

    # Every known insider is present, even ones absent from the population.
    assert insider_set <= result

    # Normals are population members that are not insiders. The sampled normals
    # are exactly the non-insider members of the result.
    normals = pop_set - insider_set
    sampled_normals = result - insider_set

    assert sampled_normals <= normals, "result contains a non-population normal"

    expected_normal_count = min(max(0, sample_size), len(normals))
    assert len(sampled_normals) == expected_normal_count

    # The result is exactly insiders plus the sampled normals, nothing else.
    assert result == insider_set | sampled_normals

    # Deterministic under a fixed seed.
    assert select_subset(pop_set, insider_set, sample_size, seed) == result
