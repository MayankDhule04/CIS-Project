"""Data_Profiler — inspect the raw CSV logs before any parser is written.

This tool opens each raw event file one slice at a time (never loading a whole
file into memory) and reports three facts about it: the column names it carries,
how many rows it contains, and the span of dates it covers. Running it first
means the ingestion parsers are derived from the data's real shape instead of
guesses. The directory it scans is read from ``config.yaml`` (Requirement 1.4),
never hard-coded.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


def _to_datetime(series: pd.Series) -> pd.Series:
    """Parse a column to datetimes, coercing failures to NaT.

    Suppresses pandas' mixed-format inference warning: the profiler accepts
    heterogeneous date strings on purpose since it does not yet know the schema.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce")

# How many rows to pull per chunk. Keeps memory flat on 20 GB files.
_CHUNK_SIZE = 100_000

# Column-name hints that suggest a column holds dates/timestamps. Used only to
# break ties; detection ultimately relies on whether values actually parse.
_DATE_NAME_HINTS = ("date", "time", "timestamp", "datetime", "ts")


@dataclass
class FileProfile:
    """A single raw CSV's discovered structure.

    Attributes:
        filename: The CSV file name (not the full path).
        columns: The column header, in file order.
        row_count: Total data rows across all chunks.
        date_range: (min_date, max_date) from the detected date column, or
            ``None`` when no date-like column was found or the file is empty.
    """

    filename: str
    columns: list[str]
    row_count: int
    date_range: Optional[tuple[date, date]]


def _resolve_raw_dir(config_path: str = "config.yaml") -> str:
    """Return the raw data directory, read from the Config_File.

    Prefers the typed loader in ``src.ueba.config`` if it is already available
    (task 1.2), and falls back to reading ``config.yaml`` directly with PyYAML
    so this script never depends on that module existing yet. The path is never
    hard-coded (Requirement 1.4).
    """
    # Defensive: use the typed config loader if it has landed.
    try:
        from src.ueba.config import load_config  # type: ignore

        cfg = load_config(config_path)
        raw_dir = getattr(cfg, "raw_data_dir", None)
        if raw_dir:
            return _make_absolute(raw_dir, config_path)
    except Exception:
        # Loader not present yet or failed — fall back to raw YAML below.
        pass

    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    raw_dir = (data.get("paths") or {}).get("raw_data_dir")
    if not raw_dir:
        raise KeyError(
            "config.yaml is missing paths.raw_data_dir; cannot locate raw data."
        )
    return _make_absolute(raw_dir, config_path)


def _make_absolute(raw_dir: str, config_path: str) -> str:
    """Resolve a (possibly relative) raw dir against the config file's folder."""
    if os.path.isabs(raw_dir):
        return raw_dir
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(config_dir, raw_dir))


def _detect_date_column(sample: pd.DataFrame) -> Optional[str]:
    """Pick the column most likely to hold dates from a sample chunk.

    The column name is never assumed. Each column is parsed as datetime and the
    one with the most successfully parsed values wins; a name hint only breaks
    ties. Returns ``None`` if no column parses into any dates.
    """
    best_col: Optional[str] = None
    best_score = 0.0

    for col in sample.columns:
        parsed = _to_datetime(sample[col])
        valid = int(parsed.notna().sum())
        if valid == 0:
            continue
        # Fraction of parseable values, nudged up for date-like names.
        score = valid / len(sample)
        if any(hint in str(col).lower() for hint in _DATE_NAME_HINTS):
            score += 1.0
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def profile_file(path: str) -> FileProfile:
    """Profile a single CSV by streaming it in chunks."""
    filename = os.path.basename(path)
    columns: list[str] = []
    row_count = 0
    date_col: Optional[str] = None
    min_date: Optional[pd.Timestamp] = None
    max_date: Optional[pd.Timestamp] = None

    for chunk in pd.read_csv(path, chunksize=_CHUNK_SIZE):
        if not columns:
            columns = list(chunk.columns)
            date_col = _detect_date_column(chunk)

        row_count += len(chunk)

        if date_col is not None and date_col in chunk.columns:
            parsed = _to_datetime(chunk[date_col]).dropna()
            if not parsed.empty:
                chunk_min = parsed.min()
                chunk_max = parsed.max()
                min_date = chunk_min if min_date is None else min(min_date, chunk_min)
                max_date = chunk_max if max_date is None else max(max_date, chunk_max)

    date_range: Optional[tuple[date, date]] = None
    if min_date is not None and max_date is not None:
        date_range = (min_date.date(), max_date.date())

    return FileProfile(
        filename=filename,
        columns=columns,
        row_count=row_count,
        date_range=date_range,
    )


def profile_directory(raw_dir: str) -> list[FileProfile]:
    """Inspect every CSV in ``raw_dir`` and print a short report per file.

    Returns one :class:`FileProfile` per CSV found. If the directory is missing
    or holds no CSVs, prints a notice and returns an empty list instead of
    raising, so the tool is safe to run on an empty workspace.
    """
    if not os.path.isdir(raw_dir):
        print(f"[profile] Raw data directory not found: {raw_dir}")
        return []

    csv_files = sorted(
        entry
        for entry in os.listdir(raw_dir)
        if entry.lower().endswith(".csv")
        and os.path.isfile(os.path.join(raw_dir, entry))
    )

    if not csv_files:
        print(f"[profile] No CSV files found in {raw_dir}")
        return []

    profiles: list[FileProfile] = []
    for name in csv_files:
        profile = profile_file(os.path.join(raw_dir, name))
        profiles.append(profile)
        _print_profile(profile)

    return profiles


def _print_profile(profile: FileProfile) -> None:
    """Print header, row count, and date range for one file (Requirement 1.3)."""
    if profile.date_range is None:
        span = "no date column detected"
    else:
        start, end = profile.date_range
        span = f"{start.isoformat()} -> {end.isoformat()}"

    print(f"=== {profile.filename} ===")
    print(f"  columns ({len(profile.columns)}): {', '.join(profile.columns)}")
    print(f"  rows: {profile.row_count:,}")
    print(f"  date range: {span}")


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point: resolve the raw dir from config, then profile it."""
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = argv[0] if argv else "config.yaml"

    raw_dir = _resolve_raw_dir(config_path)
    print(f"[profile] Scanning raw data directory: {raw_dir}")
    profile_directory(raw_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
