"""Adapters that turn the real CERT r4.2 layout into the pipeline's helpers.

The UEBA pipeline consumes two small helper files from the raw data directory:

* ``roles.csv`` — columns ``user``, ``role``; read by
  :func:`src.ueba.features.extractor._load_roles` and
  :func:`src.ueba.ingestion.subset.read_population`.
* ``labels.csv`` — columns ``user``, ``day``, ``insider``; read by
  :func:`src.ueba.evaluation.metrics.load_labels` and
  :func:`src.ueba.ingestion.subset.read_insiders`.

The synthetic generator writes both directly. The real CERT r4.2 dataset does
not — it stores roles in monthly LDAP snapshots and ships insider ground truth
separately. The two functions here build the helper files from the real
dataset's own formats so nothing downstream has to change.

Design decisions (documented so they are not surprises)
-------------------------------------------------------
* **Role over time — most-recent wins.** A user's role can change month to
  month across the LDAP snapshots. :func:`build_roles_from_ldap` resolves each
  user to the role from the *most recent* monthly file in which that user
  appears. The monthly files are named ``YYYY-MM.csv``, which sort
  chronologically as plain strings, so "most recent" is simply the last file in
  sorted order that lists the user.
* **No fabricated labels.** :func:`build_labels_from_answers` never invents
  labels. If the answers path does not exist it raises
  :class:`FileNotFoundError` with an actionable message, so a run fails loudly
  rather than silently evaluating against empty ground truth.
* **Tolerant parsing.** Both adapters accept header/case variations and parse
  CERT's ``MM/DD/YYYY HH:MM:SS`` and ``MM/DD/YYYY`` date formats robustly.

Supported answer formats
------------------------
:func:`build_labels_from_answers` accepts either a path to a single file or a
directory:

* **A published ``insiders.csv``** — columns like
  ``dataset,scenario,details,user,start,end`` where ``start``/``end`` give each
  insider's malicious date *range*. Every day in ``[start, end]`` becomes a
  positive insider user-day for that user.
* **An ``answers/`` directory** — a tree of per-scenario CSV files whose rows
  reference individual malicious log lines. Each row is mined for a user id and
  a date; every such ``(user, day)`` becomes a positive. Ranged
  ``insiders.csv``-style files found inside the tree are expanded too.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

# Logical column-name synonyms we accept in LDAP and answer files, lower-cased.
_USER_COLUMN_NAMES = ("user_id", "user", "userid", "user id")
_ROLE_COLUMN_NAMES = ("role",)
_DATE_COLUMN_NAMES = ("date", "timestamp", "time", "day")
_START_COLUMN_NAMES = ("start", "start_date", "begin")
_END_COLUMN_NAMES = ("end", "end_date", "finish")

# CERT user ids look like "CDE1846" / "ASD0577": a few letters then a few digits.
_USER_ID_PATTERN = re.compile(r"^[A-Za-z]{1,6}\d{2,6}$")

# Date formats CERT uses, tried in order before falling back to flexible parse.
_KNOWN_DATE_FORMATS = ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y")

# Output file names and schemas (must match what the pipeline already reads).
ROLES_FILENAME = "roles.csv"
LABELS_FILENAME = "labels.csv"
_ROLES_COLUMNS = ["user", "role"]
_LABELS_COLUMNS = ["user", "day", "insider"]


# ---------------------------------------------------------------------------
# Small shared helpers.
# ---------------------------------------------------------------------------


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Return the first real column whose lower-cased name is a candidate."""
    lowered = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _parse_date(value: object) -> Optional[pd.Timestamp]:
    """Parse one CERT date string into a Timestamp, or ``None`` if it cannot.

    Tries the explicit ``MM/DD/YYYY HH:MM:SS`` and ``MM/DD/YYYY`` formats first
    (the two CERT uses), then falls back to pandas' flexible parser. Returns
    ``None`` for blanks or unparseable values so callers can skip them.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in _KNOWN_DATE_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            return parsed
    parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else parsed


def _clean_user(value: object) -> Optional[str]:
    """Return a trimmed, non-empty user id string, or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# ---------------------------------------------------------------------------
# LDAP -> roles.csv
# ---------------------------------------------------------------------------


def build_roles_from_ldap(
    ldap_dir: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build a user -> role table from the monthly LDAP snapshots.

    Reads every ``*.csv`` in ``ldap_dir`` (CERT names them ``YYYY-MM.csv``) and
    resolves each user to the ``role`` from the *most recent* monthly file in
    which the user appears. Because the file names sort chronologically as
    strings, the snapshots are processed oldest-first and a later month simply
    overwrites an earlier role for the same user.

    Args:
        ldap_dir: Directory holding the monthly LDAP CSV snapshots. Each file
            must carry a user column (``user_id``/``user``) and a ``role``
            column; header case and the exact user-column spelling are tolerated.
        out_path: Optional path to write ``roles.csv`` to. When given, the
            result is written there with columns ``user``, ``role``.

    Returns:
        A DataFrame with columns ``user`` and ``role``, one row per user, sorted
        by user id.

    Raises:
        FileNotFoundError: If ``ldap_dir`` does not exist or holds no CSV files.
        ValueError: If a snapshot lacks a recognizable user or role column.
    """
    directory = Path(ldap_dir)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"LDAP directory not found: {directory}. Point --ldap-dir at the "
            f"folder of monthly LDAP snapshots (e.g. data/raw/r4.2/r4.2/LDAP)."
        )

    monthly_files = sorted(directory.glob("*.csv"))
    if not monthly_files:
        raise FileNotFoundError(
            f"No LDAP snapshot CSV files found in {directory}."
        )

    # Oldest-first: later (more recent) months overwrite earlier roles, so the
    # final value for each user is the most-recent-month role.
    user_to_role: dict[str, str] = {}
    for monthly_file in monthly_files:
        frame = pd.read_csv(monthly_file, dtype=str)
        user_col = _find_column(frame.columns, _USER_COLUMN_NAMES)
        role_col = _find_column(frame.columns, _ROLE_COLUMN_NAMES)
        if user_col is None or role_col is None:
            raise ValueError(
                f"LDAP snapshot {monthly_file.name} is missing a user or role "
                f"column (found columns: {list(frame.columns)})."
            )
        for user_value, role_value in zip(frame[user_col], frame[role_col]):
            user = _clean_user(user_value)
            if user is None:
                continue
            role = None if role_value is None else str(role_value).strip()
            user_to_role[user] = role if role else pd.NA

    roles = (
        pd.DataFrame(
            {"user": list(user_to_role.keys()), "role": list(user_to_role.values())}
        )
        .astype({"user": "string", "role": "string"})
        .sort_values("user")
        .reset_index(drop=True)
    )

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        roles.to_csv(out, index=False)

    return roles


# ---------------------------------------------------------------------------
# answers -> labels.csv
# ---------------------------------------------------------------------------


def _normalise_event_days(
    event_days: object,
) -> Optional[set[tuple[str, str]]]:
    """Coerce the optional ``event_days`` argument into a set of (user, day).

    Accepts ``None`` (no restriction), a DataFrame with ``user``/``day``
    columns, a path to a parquet/csv file with those columns, or any iterable
    of ``(user, day)`` pairs. Days are normalised to ``YYYY-MM-DD`` strings so
    they compare cleanly against expanded label days.
    """
    if event_days is None:
        return None

    frame: Optional[pd.DataFrame] = None
    if isinstance(event_days, pd.DataFrame):
        frame = event_days
    elif isinstance(event_days, (str, Path)):
        path = Path(event_days)
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)

    if frame is not None:
        users = frame["user"].astype("string").str.strip()
        days = pd.to_datetime(frame["day"]).dt.strftime("%Y-%m-%d")
        return set(zip(users.tolist(), days.tolist()))

    # Fall back to an iterable of (user, day) pairs.
    result: set[tuple[str, str]] = set()
    for user, day in event_days:  # type: ignore[misc]
        parsed = _parse_date(day) if not isinstance(day, pd.Timestamp) else day
        if parsed is None:
            continue
        result.add((str(user).strip(), parsed.strftime("%Y-%m-%d")))
    return result


def _expand_range(user: str, start: object, end: object) -> list[tuple[str, str]]:
    """Expand one insider's ``[start, end]`` into per-day ``(user, day)`` rows."""
    start_ts = _parse_date(start)
    end_ts = _parse_date(end)
    if start_ts is None or end_ts is None:
        return []
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts
    days = pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D")
    return [(user, day.strftime("%Y-%m-%d")) for day in days]


def _labels_from_dataframe(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """Extract ``(user, day)`` positives from one already-read answer frame.

    Handles three shapes, chosen by inspecting the columns:

    * ranged insiders file (user + ``start``/``end``) -> expand each range;
    * per-event file with a recognizable user and date column -> one row each;
    * otherwise treat the frame as headerless and mine every cell for a user id
      and a date.
    """
    user_col = _find_column(frame.columns, _USER_COLUMN_NAMES)
    start_col = _find_column(frame.columns, _START_COLUMN_NAMES)
    end_col = _find_column(frame.columns, _END_COLUMN_NAMES)
    date_col = _find_column(frame.columns, _DATE_COLUMN_NAMES)

    pairs: list[tuple[str, str]] = []

    if user_col is not None and start_col is not None and end_col is not None:
        for _, row in frame.iterrows():
            user = _clean_user(row[user_col])
            if user is None:
                continue
            pairs.extend(_expand_range(user, row[start_col], row[end_col]))
        return pairs

    if user_col is not None and date_col is not None:
        for user_value, date_value in zip(frame[user_col], frame[date_col]):
            user = _clean_user(user_value)
            parsed = _parse_date(date_value)
            if user is None or parsed is None:
                continue
            pairs.append((user, parsed.strftime("%Y-%m-%d")))
        return pairs

    # Headerless / unrecognized: mine each row for a user id and a date. The
    # column headers that pandas mistook for data are the first data row, so we
    # also scan the header tokens themselves.
    rows = [list(frame.columns)] + frame.astype(str).values.tolist()
    for row in rows:
        pairs.extend(_mine_row(row))
    return pairs


def _mine_row(fields: Iterable[object]) -> list[tuple[str, str]]:
    """Find a user id and a date within one raw row of unknown structure."""
    tokens = [str(field).strip() for field in fields]
    user = next((tok for tok in tokens if _USER_ID_PATTERN.match(tok)), None)
    if user is None:
        return []
    for tok in tokens:
        if "/" not in tok and "-" not in tok:
            continue
        parsed = _parse_date(tok)
        if parsed is not None:
            return [(user, parsed.strftime("%Y-%m-%d"))]
    return []


def _read_answer_file(path: Path) -> list[tuple[str, str]]:
    """Read one answer CSV and return its ``(user, day)`` positives.

    Tries a header-aware read first. If that surfaces no recognizable
    user/date/range columns — or if the file has *ragged* rows that the CSV
    parser cannot square into a table — re-reads the file with no header so the
    heuristic miner can inspect every cell. CERT's per-scenario answer files
    interleave several event types (logon, device, http, email) whose rows have
    different column counts, so the header-aware read routinely fails on them;
    those files must still be mined, not silently dropped.
    """
    headed: Optional[pd.DataFrame] = None
    try:
        headed = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return []
    except pd.errors.ParserError:
        # Ragged rows (mixed event types) — fall through to headerless mining.
        headed = None

    if headed is not None:
        has_user = _find_column(headed.columns, _USER_COLUMN_NAMES) is not None
        has_date = _find_column(headed.columns, _DATE_COLUMN_NAMES) is not None
        has_range = (
            _find_column(headed.columns, _START_COLUMN_NAMES) is not None
            and _find_column(headed.columns, _END_COLUMN_NAMES) is not None
        )
        if has_user and (has_date or has_range):
            return _labels_from_dataframe(headed)

    # No usable header (or a ragged file) -> re-read headerless and mine cells.
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            pairs: list[tuple[str, str]] = []
            for row in csv.reader(handle):
                pairs.extend(_mine_row(row))
        return pairs
    except OSError:
        return []


def _collect_pairs_from_path(path: Path) -> list[tuple[str, str]]:
    """Mine ``(user, day)`` positives from a single file or a directory tree.

    A file is read directly; a directory is walked for every ``*.csv`` beneath
    it. This is the raw collection step shared by
    :func:`build_labels_from_answers` and :func:`build_labels_for_release`.
    """
    pairs: list[tuple[str, str]] = []
    if path.is_file():
        pairs.extend(_read_answer_file(path))
    else:
        for csv_file in sorted(path.rglob("*.csv")):
            pairs.extend(_read_answer_file(csv_file))
    return pairs


def _pairs_to_labels(
    pairs: list[tuple[str, str]],
    event_days: object = None,
    out_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Turn mined ``(user, day)`` pairs into the internal labels table.

    De-duplicates and sorts the positives, optionally narrows them to user-days
    that actually have events (``event_days``), stamps ``insider = 1``, and —
    when ``out_path`` is given — writes ``labels.csv`` with the exact columns
    the pipeline reads (``user``, ``day``, ``insider``).
    """
    allowed = _normalise_event_days(event_days)
    if allowed is not None:
        pairs = [pair for pair in pairs if pair in allowed]

    if pairs:
        labels = (
            pd.DataFrame(pairs, columns=["user", "day"])
            .drop_duplicates()
            .sort_values(["user", "day"])
            .reset_index(drop=True)
        )
        labels["insider"] = 1
    else:
        labels = pd.DataFrame({"user": [], "day": [], "insider": []})

    labels = labels.astype({"user": "string", "day": "string", "insider": "int64"})
    labels = labels[_LABELS_COLUMNS]

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(out, index=False)

    return labels


def build_labels_from_answers(
    answers_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    event_days: object = None,
) -> pd.DataFrame:
    """Build ``labels.csv`` ground truth from CERT answers.

    Converts either a published ``insiders.csv`` (per-user malicious date
    ranges) or an ``answers/`` directory (per-scenario files of malicious log
    lines) into the internal label table: one row per positive insider
    ``(user, day)`` with ``insider = 1``.

    .. warning::
        Pointed at a *multi-release* answers tree (CERT ships every release's
        answers side by side), this walks **all** ``*.csv`` beneath the path and
        therefore mixes releases. To restrict to a single CERT release — e.g.
        the r4.2 event logs this pipeline scores — use
        :func:`build_labels_for_release` instead, which selects only that
        release's scenario directories.

    Args:
        answers_path: Path to a single answers file (e.g. ``insiders.csv``) or a
            directory tree of answer files. Must exist.
        out_path: Optional path to write ``labels.csv`` to, with columns
            ``user``, ``day``, ``insider``.
        event_days: Optional restriction of positives to user-days that actually
            have events. May be ``None`` (no restriction), a DataFrame with
            ``user``/``day`` columns, a path to a parquet/csv with those columns
            (e.g. ``user_day.parquet``), or an iterable of ``(user, day)`` pairs.
            When given, only expanded ``(user, day)`` positives that also appear
            in ``event_days`` are kept — useful because an insider's date range
            can span days the user had no logged activity.

    Returns:
        A DataFrame with columns ``user`` (string), ``day`` (``YYYY-MM-DD``
        string), and ``insider`` (int ``1``), de-duplicated and sorted.

    Raises:
        FileNotFoundError: If ``answers_path`` does not exist. Labels are never
            fabricated when the ground truth is absent.
    """
    path = Path(answers_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Answers path not found: {path}. CERT r4.2 ground truth ships "
            f"separately — provide the extracted answers directory or the "
            f"published insiders.csv. Labels are not fabricated when absent."
        )

    pairs = _collect_pairs_from_path(path)
    return _pairs_to_labels(pairs, event_days=event_days, out_path=out_path)


def _insider_from_filename(path: Path) -> Optional[str]:
    """Extract the insider user id from a per-insider answer file name.

    CERT names each per-scenario file ``{release}-{scenario}-{USERID}.csv`` —
    e.g. ``r4.2-1-AAM0658.csv`` → ``AAM0658``. The user id is the final
    hyphen-separated token of the stem and must look like a CERT id. Returns
    ``None`` when the name does not carry a recognizable trailing user id (for
    those files no filename restriction is applied).
    """
    token = path.stem.split("-")[-1].strip()
    return token if _USER_ID_PATTERN.match(token) else None


def collect_release_scenario_dirs(
    answers_root: Union[str, Path], release: str
) -> list[Path]:
    """Return the per-scenario answer directories for one CERT release.

    CERT's published ``answers/`` tree stores every release's ground truth side
    by side: single-file answers (``r2.csv``, ``r4.1-1.csv``, …) and, for the
    releases with per-insider log-line files, per-scenario *directories*
    (``r4.2-1/``, ``r4.2-2/``, ``r4.2-3/`` for release 4.2). This returns only
    the scenario directories whose name begins with ``"{release}-"`` — e.g.
    ``release="r4.2"`` selects ``r4.2-1``, ``r4.2-2``, ``r4.2-3`` while skipping
    ``r4.1-*``, ``r5.2-*``, and every other release.

    Matching is anchored to ``"{release}-"`` (with the trailing hyphen) so a
    request for ``r4.2`` never also grabs a hypothetical ``r4.20-*`` directory.

    Args:
        answers_root: The CERT ``answers/`` directory holding all releases.
        release: The release prefix to keep, e.g. ``"r4.2"``.

    Returns:
        The matching scenario directories, sorted by name. May be empty if the
        release ships no per-scenario directories under ``answers_root``.

    Raises:
        FileNotFoundError: If ``answers_root`` is not an existing directory.
    """
    root = Path(answers_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Answers directory not found: {root}. Point it at the CERT "
            f"answers/ folder that holds the per-release ground truth."
        )
    prefix = f"{release}-"
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and child.name.startswith(prefix)
    )


def build_labels_for_release(
    answers_root: Union[str, Path],
    release: str = "r4.2",
    out_path: Optional[Union[str, Path]] = None,
    event_days: object = None,
) -> pd.DataFrame:
    """Build ``labels.csv`` from **one** CERT release's answers only.

    The real CERT ``answers/`` tree interleaves every release (r2, r3.x, r4.1,
    r4.2, r5.x, r6.x). Mining the whole tree would wrongly label users from
    unrelated releases against the r4.2 event logs. This function keeps only the
    scenario directories for ``release`` (see
    :func:`collect_release_scenario_dirs`) — for ``r4.2`` those are the
    ``r4.2-1``, ``r4.2-2``, ``r4.2-3`` folders of per-insider malicious
    log-line files — and mines the exact malicious ``(user, day)`` positives
    from them.

    This is the precise, per-day ground truth: each positive is a day the
    insider actually performed malicious activity, not merely a day inside a
    published date range.

    Each per-insider file is named for its insider
    (``r4.2-1-AAM0658.csv`` → ``AAM0658``). Only rows whose mined user matches
    that filename insider are kept, so a correspondent who merely appears in the
    insider's malicious email thread (CERT's scenario-3 files include both sides
    of the exchange) is never mislabelled as an insider.

    Args:
        answers_root: The CERT ``answers/`` directory holding all releases.
        release: The release prefix to keep (default ``"r4.2"``).
        out_path: Optional path to write ``labels.csv`` to.
        event_days: Optional restriction to user-days with real events, passed
            through to :func:`_pairs_to_labels` (see
            :func:`build_labels_from_answers`).

    Returns:
        A DataFrame with columns ``user``, ``day`` (``YYYY-MM-DD``), ``insider``
        (``1``), containing only ``release`` insiders' malicious user-days.

    Raises:
        FileNotFoundError: If ``answers_root`` does not exist, or the release
            ships no scenario directories beneath it (so nothing is labelled and
            an empty ground truth is never silently produced).
    """
    scenario_dirs = collect_release_scenario_dirs(answers_root, release)
    if not scenario_dirs:
        raise FileNotFoundError(
            f"No per-scenario answer directories for release '{release}' under "
            f"{Path(answers_root)}. Expected folders like '{release}-1', "
            f"'{release}-2' of per-insider malicious log-line files."
        )

    pairs: list[tuple[str, str]] = []
    for scenario_dir in scenario_dirs:
        for csv_file in sorted(scenario_dir.rglob("*.csv")):
            insider = _insider_from_filename(csv_file)
            file_pairs = _read_answer_file(csv_file)
            if insider is not None:
                # Keep only the file's own insider; drop email-thread
                # correspondents and any stray ids mined from the rows.
                file_pairs = [pair for pair in file_pairs if pair[0] == insider]
            pairs.extend(file_pairs)

    return _pairs_to_labels(pairs, event_days=event_days, out_path=out_path)
