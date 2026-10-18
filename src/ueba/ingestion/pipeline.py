"""Ingestion_Pipeline — turn a mountain of raw events into one row per user-day.

The raw CERT 4.2 logs (or their synthetic stand-ins) are large: millions of
individual logon, device, web, and email events spread across four CSV files.
This module reads each of those files *a slice at a time* — never loading a
whole file into memory — and boils them down to a compact table with exactly
one row per ``(user, calendar_day)``. That compact table, ``user_day.parquet``,
is what every later stage of the pipeline builds on.

What ``ingest`` does, in plain English:

* Opens each raw CSV (``logon``, ``device``, ``http``, ``email``) and reads it
  in fixed-size chunks using pandas' ``chunksize``. No file is ever read whole.
* Uses the column map from ``config`` to find the user and timestamp columns,
  so no source column name is hard-coded here — the mapping comes from config.
* Parses each timestamp down to its calendar day and throws away any event
  whose user id is null or blank *before* anything is counted, so dropped rows
  can never affect a surviving user-day's totals.
* Aggregates to one row per user per active day, carrying a basic per-source
  event count for that day (``logon_events``, ``device_events``,
  ``http_events``, ``email_events``, and their ``total_events`` sum).
* Attaches each user's role from ``roles.csv`` so the later peer-baseline stage
  has a group to compare against.
* Writes the result to ``<processed_dir>/user_day.parquet`` and returns it.

Scope and clean hand-offs:

* **Subset selection (task 2.3)** is deliberately *not* implemented here.
  ``ingest`` accepts an optional ``users`` set and, when given, restricts its
  output to those users — that is the single hook the subset-selection task
  plugs into. This module never samples or decides the subset itself.
* **Full behavioral features (task 3.1, Feature_Extractor)** are also not
  computed here. Ingestion produces the normalized, de-duplicated per-user-day
  skeleton — correct users, correct days, no null users, one row each, plus
  role and coarse event counts. The Feature_Extractor enriches this skeleton
  with the detailed columns (after-hours logons, distinct machines, site
  categories, and the rest), which need event-level detail and config windows.

Keeping those two concerns out means this module has one job and does it
predictably: a correct, compact, round-trippable per-user-day table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..config import Config

# The raw event files this pipeline knows how to read, in a stable order.
# Each maps to the per-day count column it contributes to the output table.
EVENT_FILES: dict[str, str] = {
    "logon": "logon_events",
    "device": "device_events",
    "http": "http_events",
    "email": "email_events",
}

# CERT release 4.2 timestamps look like "01/04/2010 08:37:12" (month/day/year).
# We try this explicit format first and fall back to flexible parsing only if
# it clearly does not fit, which keeps parsing both fast and forgiving.
KNOWN_TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"

# Default rows-per-chunk. Large enough to be efficient, small enough that a
# single chunk never approaches the size of a whole raw file.
DEFAULT_CHUNK_SIZE = 250_000

# The name of the artifact this pipeline writes.
USER_DAY_FILENAME = "user_day.parquet"

# Helper file (shipped alongside the event logs) mapping user -> role.
ROLES_FILENAME = "roles.csv"


def _parse_days(raw_timestamps: pd.Series) -> pd.Series:
    """Parse a column of raw timestamp strings down to calendar days.

    Returns a ``datetime64[ns]`` series normalised to midnight (the calendar
    day). Values that cannot be parsed become ``NaT`` so the caller can drop
    them. The known CERT format is tried first; if most values do not match it
    (a differently formatted dataset), we retry with pandas' flexible parser.
    """
    parsed = pd.to_datetime(
        raw_timestamps, format=KNOWN_TIMESTAMP_FORMAT, errors="coerce"
    )
    # If the fixed format fit almost nothing, the data uses another layout.
    if len(parsed) > 0 and parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(raw_timestamps, errors="coerce")
    return parsed.dt.normalize()


def _clean_users(raw_users: pd.Series) -> pd.Series:
    """Normalise user ids to trimmed strings, with blanks marked as missing.

    Null users and users that are empty once whitespace is stripped are turned
    into ``<NA>`` so a single downstream ``notna``/non-empty filter removes
    them. This is where Requirement 2.5 (exclude null-user events) is enforced.
    """
    users = raw_users.astype("string").str.strip()
    # Treat empty strings as missing so they are dropped alongside true nulls.
    return users.mask(users == "")


def _aggregate_event_file(
    path: Path, user_col: str, date_col: str, count_col: str, chunk_size: int
) -> pd.DataFrame:
    """Chunk-read one event CSV and return per-``(user, day)`` event counts.

    Only the user and timestamp columns are loaded, and only ``chunk_size``
    rows at a time, so the raw file is never held whole in memory. Each chunk
    is cleaned (null/blank users and unparseable days dropped) and pre-
    aggregated; the small per-chunk aggregates are then summed together.
    """
    # Restrict the read to just the two columns we need, as strings, so a wide
    # file (e.g. email with 11 columns) never pulls unused data into memory.
    wanted = {user_col, date_col}
    partial_counts: list[pd.DataFrame] = []

    reader = pd.read_csv(
        path,
        usecols=lambda column: column in wanted,
        dtype=str,
        chunksize=chunk_size,
    )
    for chunk in reader:
        if user_col not in chunk.columns or date_col not in chunk.columns:
            # The configured column is absent from this file; nothing to count.
            continue
        frame = pd.DataFrame(
            {
                "user": _clean_users(chunk[user_col]),
                "day": _parse_days(chunk[date_col]),
            }
        )
        # Drop null/blank users and unparseable days before any counting.
        frame = frame[frame["user"].notna() & frame["day"].notna()]
        if frame.empty:
            continue
        chunk_counts = (
            frame.groupby(["user", "day"], sort=False)
            .size()
            .rename(count_col)
            .reset_index()
        )
        partial_counts.append(chunk_counts)

    if not partial_counts:
        return pd.DataFrame(columns=["user", "day", count_col])

    # Sum the per-chunk aggregates so events for the same user-day that landed
    # in different chunks are combined into a single total.
    combined = pd.concat(partial_counts, ignore_index=True)
    return (
        combined.groupby(["user", "day"], as_index=False, sort=False)[count_col]
        .sum()
    )


def _load_roles(raw_dir: Path) -> pd.DataFrame:
    """Load the user -> role lookup from ``roles.csv`` if it is present.

    Returns a two-column frame (``user``, ``role``). When the file is missing
    the frame is empty, and roles simply come through as missing in the output
    rather than causing ingestion to fail.
    """
    roles_path = raw_dir / ROLES_FILENAME
    if not roles_path.is_file():
        return pd.DataFrame(columns=["user", "role"])
    roles = pd.read_csv(roles_path, dtype=str)
    if "user" not in roles.columns or "role" not in roles.columns:
        return pd.DataFrame(columns=["user", "role"])
    roles = roles[["user", "role"]].copy()
    roles["user"] = roles["user"].astype("string").str.strip()
    roles = roles[roles["user"].notna() & (roles["user"] != "")]
    return roles.drop_duplicates(subset="user")


def ingest(
    config: Config,
    users: Optional[Iterable[str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    """Read every raw CSV in chunks and build the per-user-day table.

    Args:
        config: The loaded :class:`~src.ueba.config.Config`. Supplies the raw
            and processed directories and the column map used to resolve the
            user and timestamp columns for each file type.
        users: Optional iterable of user ids to restrict the output to. This is
            the integration hook for subset selection (task 2.3); when ``None``
            every user found in the raw logs is kept. Subset *sampling* is not
            performed here — only the restriction to an already-chosen set.
        chunk_size: Rows read per chunk. The default keeps any single chunk far
            smaller than a whole raw file.

    Returns:
        A DataFrame with one row per ``(user, day)`` carrying ``role`` and the
        per-source event counts, which is also written to
        ``<processed_dir>/user_day.parquet``.
    """
    raw_dir = Path(config.raw_data_dir)
    processed_dir = Path(config.processed_dir)

    # Aggregate each event file independently, then join on (user, day).
    per_file: list[pd.DataFrame] = []
    for file_type, count_col in EVENT_FILES.items():
        file_path = raw_dir / f"{file_type}.csv"
        if not file_path.is_file():
            # A missing source file simply contributes no counts.
            continue
        column_mapping = config.column_map.get(file_type, {})
        user_col = column_mapping.get("user")
        date_col = column_mapping.get("date")
        if not user_col or not date_col:
            # Without a user and date mapping we cannot place events in a day.
            continue
        per_file.append(
            _aggregate_event_file(
                file_path, user_col, date_col, count_col, chunk_size
            )
        )

    user_day = _merge_event_counts(per_file)
    user_day = _attach_roles(user_day, _load_roles(raw_dir))

    if users is not None:
        # Subset hook: keep only the pre-selected users (task 2.3 supplies them).
        keep = {str(u) for u in users}
        user_day = user_day[user_day["user"].isin(keep)].reset_index(drop=True)

    user_day = _finalise(user_day)
    _write_user_day(user_day, processed_dir)
    return user_day


def _merge_event_counts(per_file: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join the per-file counts into one row per active user-day.

    A user-day appears in the result if it had at least one event in any source
    file. Missing counts (a day with logons but no emails, say) become 0.
    """
    all_columns = ["user", "day", *EVENT_FILES.values()]
    if not per_file:
        return pd.DataFrame(columns=all_columns)

    merged: Optional[pd.DataFrame] = None
    for frame in per_file:
        if frame.empty:
            continue
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["user", "day"], how="outer")

    if merged is None:
        return pd.DataFrame(columns=all_columns)

    # Ensure every count column exists even if its source file was absent.
    for count_col in EVENT_FILES.values():
        if count_col not in merged.columns:
            merged[count_col] = 0
    return merged


def _attach_roles(user_day: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """Left-join each user's role onto the user-day table."""
    if user_day.empty:
        user_day = user_day.copy()
        user_day["role"] = pd.Series(dtype="string")
        return user_day
    if roles.empty:
        user_day = user_day.copy()
        user_day["role"] = pd.NA
        return user_day
    return user_day.merge(roles, on="user", how="left")


def _finalise(user_day: pd.DataFrame) -> pd.DataFrame:
    """Tidy types, fill count gaps, add ``total_events``, and order the rows.

    Guarantees the output invariants: exactly one row per ``(user, day)``,
    integer counts with no missing values, and a stable column order with
    ``user``, ``day``, and ``role`` first.
    """
    count_cols = list(EVENT_FILES.values())
    ordered_cols = ["user", "day", "role", *count_cols, "total_events"]

    if user_day.empty:
        empty = pd.DataFrame(columns=ordered_cols)
        empty["user"] = empty["user"].astype("string")
        empty["day"] = pd.to_datetime(empty["day"])
        empty["role"] = empty["role"].astype("string")
        for col in [*count_cols, "total_events"]:
            empty[col] = empty[col].astype("int64")
        return empty

    user_day = user_day.copy()
    user_day["user"] = user_day["user"].astype("string")
    user_day["day"] = pd.to_datetime(user_day["day"])
    user_day["role"] = (
        user_day["role"].astype("string") if "role" in user_day else pd.NA
    )

    for col in count_cols:
        user_day[col] = (
            pd.to_numeric(user_day[col], errors="coerce").fillna(0).astype("int64")
        )
    user_day["total_events"] = user_day[count_cols].sum(axis=1).astype("int64")

    # One row per user-day is already guaranteed by the groupby/outer-join, but
    # we assert it defensively and present the rows in a deterministic order.
    user_day = (
        user_day.drop_duplicates(subset=["user", "day"])
        .sort_values(["user", "day"])
        .reset_index(drop=True)
    )
    return user_day[ordered_cols]


def _write_user_day(user_day: pd.DataFrame, processed_dir: Path) -> Path:
    """Write the user-day table to ``<processed_dir>/user_day.parquet``."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / USER_DAY_FILENAME
    user_day.to_parquet(out_path, engine="pyarrow", index=False)
    return out_path


if __name__ == "__main__":  # pragma: no cover - manual convenience entry point
    from ..config import load_config

    cfg = load_config()
    result = ingest(cfg)
    print(
        f"[ingest] Wrote {len(result)} user-day rows to "
        f"{Path(cfg.processed_dir) / USER_DAY_FILENAME}"
    )
