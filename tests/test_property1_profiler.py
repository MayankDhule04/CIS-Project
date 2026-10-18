"""Property-based tests for the Data_Profiler.

Feature: ueba-insider-threat-detection, Property 1: Data_Profiler reports match
ground truth.

For any raw CSV with an arbitrary header, row set, and date column, the
Data_Profiler must report a row count equal to the true number of rows, a column
list equal to the true header, and a date range equal to the true minimum and
maximum dates (design.md Property 1; Requirement 1.3).
"""

from __future__ import annotations

import os
import string
import tempfile
from datetime import date

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import profile_data

# Column-name substrings the profiler treats as date hints. Non-date columns
# must avoid these so date-column detection stays unambiguous in the test.
_DATE_NAME_HINTS = ("date", "time", "timestamp", "datetime", "ts")

# A clearly date-named column so the profiler's name hint makes it the
# unambiguous winner of date-column detection regardless of other columns.
_DATE_COL = "event_date"

# pandas Timestamps only cover roughly 1677-2262; stay well inside that window
# so every written date parses back (no silent NaT that would skew min/max).
_MIN_DATE = date(1900, 1, 1)
_MAX_DATE = date(2100, 12, 31)


def _has_hint(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _DATE_NAME_HINTS)


# Names for the non-date columns: lowercase letters only, never containing a
# date-name hint, so only _DATE_COL is detected as the date column.
_plain_names = st.text(
    alphabet=string.ascii_lowercase, min_size=1, max_size=8
).filter(lambda s: not _has_hint(s))

# Non-date cell values: short lowercase strings. Even if pandas coerces some to
# NaN/NaT, they never out-score the hinted date column.
_plain_values = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6)


@st.composite
def csv_cases(draw):
    """Build an arbitrary CSV layout with a single, clear date column.

    Returns ``(header, dates, other_data)`` where ``header`` is the full column
    order written to disk, ``dates`` is the ground-truth date column, and
    ``other_data`` maps each non-date column name to its values.
    """
    n_rows = draw(st.integers(min_value=1, max_value=25))

    dates = draw(
        st.lists(
            st.dates(min_value=_MIN_DATE, max_value=_MAX_DATE),
            min_size=n_rows,
            max_size=n_rows,
        )
    )

    other_names = draw(
        st.lists(_plain_names, min_size=0, max_size=4, unique=True)
    )

    other_data = {
        name: draw(
            st.lists(_plain_values, min_size=n_rows, max_size=n_rows)
        )
        for name in other_names
    }

    # Place the date column at an arbitrary position in the header.
    header = list(other_names)
    pos = draw(st.integers(min_value=0, max_value=len(header)))
    header.insert(pos, _DATE_COL)

    return header, dates, other_data


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(case=csv_cases())
def test_profiler_reports_match_ground_truth(case):
    """Property 1: reported row count, columns, and date range equal the truth."""
    header, dates, other_data = case

    # Assemble the DataFrame and write it as a real CSV on disk.
    data = dict(other_data)
    data[_DATE_COL] = [d.isoformat() for d in dates]
    frame = pd.DataFrame(data)[header]

    # Fresh temp dir per generated example so examples never interfere.
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "events.csv")
        frame.to_csv(csv_path, index=False)

        profile = profile_data.profile_file(csv_path)

    # Ground truth derived from exactly what was written.
    expected_columns = header
    expected_row_count = len(dates)
    expected_date_range = (min(dates), max(dates))

    assert profile.columns == expected_columns
    assert profile.row_count == expected_row_count
    assert profile.date_range == expected_date_range
