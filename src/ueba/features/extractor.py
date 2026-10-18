"""Feature_Extractor — turn each person's raw daily activity into numbers.

Ingestion (see :mod:`src.ueba.ingestion.pipeline`) produces a *coarse*
per-user-day skeleton: for each ``(user, calendar_day)`` it records how many
events of each kind happened. That is enough to know *who was active when*, but
not enough to score behaviour. This module fills in the detail. It reads the
raw event CSVs again — a slice at a time, never a whole file — and computes the
full behavioural feature table described in the design's Data Models section:
one row per ``(user, day)`` with four families of signals.

* **Logon family** (Req 4.1): how many times the user logged on, how many of
  those logons were outside working hours, how many landed on a weekend, and
  how many distinct machines they touched.
* **USB / removable-media family** (Req 4.2): how many removable devices were
  connected, how many of those connects were after hours, and how many files
  were copied to removable media.
* **Web family** (Req 4.3): visits to configured job sites, visits to
  configured upload/leak sites, file uploads, the number of distinct domains
  seen, and the total number of web requests.
* **Email family** (Req 4.4): emails sent, how many recipients were external to
  the organisation, and the total attachment count.

Two things are classified **strictly from config** (Req 4.5), never hard-coded:

* *After-hours* means the event's hour falls outside the half-open working-hours
  window ``[start_hour, end_hour)`` read from ``config.working_hours``.
* *Site category* means the request's domain appears in one of the configured
  lists (``config.site_categories.job_sites`` /
  ``config.site_categories.upload_leak_sites``). The weekend test uses
  ``config.weekend_days`` (Mon=0).

The result is written to ``<processed_dir>/user_day.parquet``. Because the full
feature schema is a superset of ingestion's coarse table, this file supersedes
the coarse artifact — every later stage (baselines, scoring, dashboard) reads
these columns. A user-day row exists whenever the user had *any* event that day
in *any* source; a feature with no contributing events for that day is simply 0.

This module reads raw files directly (rather than taking a pre-joined frame)
because the detailed features need event-level fields — timestamps, machine
ids, URLs, recipient lists — that the coarse per-day table has already summed
away. It reuses the same chunked-read and config-driven column resolution as
ingestion so no source column name is hard-coded here either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..config import Config

# CERT 4.2 timestamps look like "01/04/2010 08:37:12" (month/day/year). Try the
# explicit format first and fall back to flexible parsing only if it clearly
# does not fit — the same strategy ingestion uses.
KNOWN_TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"

# Default rows-per-chunk, matching ingestion: big enough to be efficient, small
# enough that a single chunk never approaches a whole raw file.
DEFAULT_CHUNK_SIZE = 250_000

# The artifact this extractor writes (the full-schema user-day table).
USER_DAY_FILENAME = "user_day.parquet"

# Helper file mapping user -> role, shipped alongside the event logs.
ROLES_FILENAME = "roles.csv"

# Activity values from the CERT schema. Compared case-insensitively so a
# differently-cased dataset still matches.
LOGON_ACTIVITY = "logon"
USB_CONNECT_ACTIVITY = "connect"

# Organisation-internal mail domain. Any recipient whose domain differs is
# counted as external. Overridable via the ``internal_domain`` argument (or an
# optional ``internal_domain`` attribute on the config object) so the default
# never has to be edited in code.
DEFAULT_INTERNAL_DOMAIN = "dtaa.com"

# The full ordered list of feature columns, exactly matching design.md's
# user_day.parquet Data Model (logon, USB, web, email families).
FEATURE_COLUMNS: list[str] = [
    # logon family (Req 4.1)
    "logon_count",
    "after_hours_logon_count",
    "weekend_logon_count",
    "distinct_machines",
    # USB / removable-media family (Req 4.2)
    "usb_connect_count",
    "after_hours_usb_count",
    "files_to_removable",
    # web family (Req 4.3)
    "job_site_visits",
    "upload_leak_site_visits",
    "file_upload_count",
    "distinct_domains",
    "total_web_requests",
    # email family (Req 4.4)
    "emails_sent",
    "external_recipients",
    "email_attachments",
]

OUTPUT_COLUMNS: list[str] = ["user", "day", "role", *FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Small shared parsing helpers (mirrors of ingestion's, kept local so the
# feature layer does not depend on ingestion's private functions).
# ---------------------------------------------------------------------------


def _parse_datetimes(raw_timestamps: pd.Series) -> pd.Series:
    """Parse raw timestamp strings into full ``datetime64`` values.

    The known CERT format is tried first; if it fits almost nothing (a
    differently formatted dataset) we retry with pandas' flexible parser.
    Unparseable values become ``NaT`` so the caller can drop them. Unlike
    ingestion we keep the *time*, not just the day, because after-hours
    classification needs the hour.
    """
    parsed = pd.to_datetime(
        raw_timestamps, format=KNOWN_TIMESTAMP_FORMAT, errors="coerce"
    )
    if len(parsed) > 0 and parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(raw_timestamps, errors="coerce")
    return parsed


def _clean_users(raw_users: pd.Series) -> pd.Series:
    """Normalise user ids to trimmed strings, with blanks marked as missing.

    Null users and users that are empty once whitespace is stripped become
    ``<NA>`` so a single downstream ``notna`` filter removes them — the same
    null-user exclusion ingestion applies (Req 2.5).
    """
    users = raw_users.astype("string").str.strip()
    return users.mask(users == "")


def _split_url(urls: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split raw URLs into a lowercased domain and the remaining path.

    ``http://job-hunt.org/upload`` becomes domain ``job-hunt.org`` and path
    ``upload``. The scheme is stripped, the text up to the first ``/`` is the
    domain, and whatever follows is the path (empty when there is none).
    """
    text = urls.astype("string").fillna("")
    no_scheme = text.str.replace(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", regex=True)
    parts = no_scheme.str.split("/", n=1)
    domain = parts.str[0].str.lower()
    path = parts.str[1].fillna("")
    return domain, path


def _after_hours_mask(hours: pd.Series, start_hour: int, end_hour: int) -> pd.Series:
    """True where an hour falls outside the half-open window [start, end)."""
    return (hours < start_hour) | (hours >= end_hour)


def _open_reader(path: Path, wanted: set[str], chunk_size: int):
    """Open a chunked CSV reader that loads only the requested columns as str."""
    return pd.read_csv(
        path,
        usecols=lambda column: column in wanted,
        dtype=str,
        chunksize=chunk_size,
    )


def _sum_parts(parts: list[pd.DataFrame], value_cols: list[str]) -> pd.DataFrame:
    """Sum additive per-chunk aggregates into one row per ``(user, day)``.

    Features that are counts (logons, emails, uploads, ...) combine by addition,
    so per-chunk partials — even when the same user-day straddled two chunks —
    are simply summed together.
    """
    if not parts:
        return pd.DataFrame(columns=["user", "day", *value_cols])
    combined = pd.concat(parts, ignore_index=True)
    return combined.groupby(["user", "day"], as_index=False, sort=False)[
        value_cols
    ].sum()


def _distinct_count(
    parts: list[pd.DataFrame], value_col: str, out_col: str
) -> pd.DataFrame:
    """Count distinct values per ``(user, day)`` across all chunks.

    Distinct counts (distinct machines, distinct domains) cannot be summed
    across chunks, so each chunk contributes its unique
    ``(user, day, value)`` triples; these are concatenated, de-duplicated once,
    and counted — giving the correct distinct total even when a user-day's
    events were split across chunks.
    """
    if not parts:
        return pd.DataFrame(columns=["user", "day", out_col])
    combined = pd.concat(parts, ignore_index=True).drop_duplicates()
    out = (
        combined.groupby(["user", "day"], as_index=False, sort=False)
        .size()
        .rename(columns={"size": out_col})
    )
    return out


# ---------------------------------------------------------------------------
# Per-source feature extractors. Each reads one raw CSV in chunks and returns a
# per-(user, day) frame carrying that source's feature columns.
# ---------------------------------------------------------------------------


def _extract_logon(
    path: Path, mapping: dict[str, str], config: Config, chunk_size: int
) -> pd.DataFrame:
    """Compute the logon family per user-day from ``logon.csv`` (Req 4.1).

    ``logon_count`` counts Logon activities; ``after_hours_logon_count`` those
    outside the configured working-hours window; ``weekend_logon_count`` those
    on configured weekend days; ``distinct_machines`` the number of distinct
    machines the user appeared on that day.
    """
    user_col = mapping.get("user")
    date_col = mapping.get("date")
    pc_col = mapping.get("pc")
    activity_col = mapping.get("activity")
    if not user_col or not date_col or not activity_col:
        return pd.DataFrame(
            columns=["user", "day", "logon_count", "after_hours_logon_count",
                     "weekend_logon_count", "distinct_machines"]
        )

    start_hour = config.working_hours.start_hour
    end_hour = config.working_hours.end_hour
    weekend_days = set(config.weekend_days)

    wanted = {user_col, date_col, activity_col}
    if pc_col:
        wanted.add(pc_col)

    count_parts: list[pd.DataFrame] = []
    machine_parts: list[pd.DataFrame] = []

    for chunk in _open_reader(path, wanted, chunk_size):
        if not {user_col, date_col, activity_col}.issubset(chunk.columns):
            continue
        parsed = _parse_datetimes(chunk[date_col])
        frame = pd.DataFrame(
            {
                "user": _clean_users(chunk[user_col]),
                "day": parsed.dt.normalize(),
                "hour": parsed.dt.hour,
                "dow": parsed.dt.dayofweek,
                "activity": chunk[activity_col].astype("string").str.strip().str.lower(),
            }
        )
        if pc_col and pc_col in chunk.columns:
            frame["pc"] = chunk[pc_col].astype("string").str.strip()
        else:
            frame["pc"] = pd.NA
        frame = frame[frame["user"].notna() & frame["day"].notna()]
        if frame.empty:
            continue

        logons = frame[frame["activity"] == LOGON_ACTIVITY]
        if not logons.empty:
            after = _after_hours_mask(logons["hour"], start_hour, end_hour)
            weekend = logons["dow"].isin(weekend_days)
            agg = pd.DataFrame(
                {
                    "user": logons["user"].values,
                    "day": logons["day"].values,
                    "logon_count": 1,
                    "after_hours_logon_count": after.astype("int64").values,
                    "weekend_logon_count": weekend.astype("int64").values,
                }
            )
            count_parts.append(
                agg.groupby(["user", "day"], as_index=False, sort=False)[
                    ["logon_count", "after_hours_logon_count", "weekend_logon_count"]
                ].sum()
            )

        # Distinct machines: unique (user, day, pc) across all logon rows.
        machines = frame.loc[frame["pc"].notna(), ["user", "day", "pc"]]
        machines = machines[machines["pc"].astype("string").str.len() > 0]
        if not machines.empty:
            machine_parts.append(machines.drop_duplicates())

    counts = _sum_parts(
        count_parts,
        ["logon_count", "after_hours_logon_count", "weekend_logon_count"],
    )
    machines = _distinct_count(machine_parts, "pc", "distinct_machines")
    if counts.empty and machines.empty:
        return pd.DataFrame(
            columns=["user", "day", "logon_count", "after_hours_logon_count",
                     "weekend_logon_count", "distinct_machines"]
        )
    return counts.merge(machines, on=["user", "day"], how="outer")


def _extract_device(
    path: Path, mapping: dict[str, str], config: Config, chunk_size: int
) -> pd.DataFrame:
    """Compute USB connect counts per user-day from ``device.csv`` (Req 4.2).

    ``usb_connect_count`` counts Connect activities; ``after_hours_usb_count``
    those outside the configured working-hours window.
    """
    user_col = mapping.get("user")
    date_col = mapping.get("date")
    activity_col = mapping.get("activity")
    if not user_col or not date_col or not activity_col:
        return pd.DataFrame(
            columns=["user", "day", "usb_connect_count", "after_hours_usb_count"]
        )

    start_hour = config.working_hours.start_hour
    end_hour = config.working_hours.end_hour

    wanted = {user_col, date_col, activity_col}
    count_parts: list[pd.DataFrame] = []

    for chunk in _open_reader(path, wanted, chunk_size):
        if not wanted.issubset(chunk.columns):
            continue
        parsed = _parse_datetimes(chunk[date_col])
        frame = pd.DataFrame(
            {
                "user": _clean_users(chunk[user_col]),
                "day": parsed.dt.normalize(),
                "hour": parsed.dt.hour,
                "activity": chunk[activity_col].astype("string").str.strip().str.lower(),
            }
        )
        frame = frame[frame["user"].notna() & frame["day"].notna()]
        connects = frame[frame["activity"] == USB_CONNECT_ACTIVITY]
        if connects.empty:
            continue
        after = _after_hours_mask(connects["hour"], start_hour, end_hour)
        agg = pd.DataFrame(
            {
                "user": connects["user"].values,
                "day": connects["day"].values,
                "usb_connect_count": 1,
                "after_hours_usb_count": after.astype("int64").values,
            }
        )
        count_parts.append(
            agg.groupby(["user", "day"], as_index=False, sort=False)[
                ["usb_connect_count", "after_hours_usb_count"]
            ].sum()
        )

    return _sum_parts(count_parts, ["usb_connect_count", "after_hours_usb_count"])


def _extract_file(
    path: Path, mapping: dict[str, str], config: Config, chunk_size: int
) -> pd.DataFrame:
    """Compute ``files_to_removable`` per user-day from ``file.csv`` (Req 4.2).

    The synthetic generator does not emit ``file.csv``, so this source is
    usually absent and the feature defaults to 0 everywhere. When the file *is*
    present and ``config.column_map`` maps a ``file`` entry, each file event is
    counted; if a ``to_removable`` logical column is mapped, only events whose
    flag is truthy are counted (a file copied *to* removable media).
    """
    user_col = mapping.get("user")
    date_col = mapping.get("date")
    removable_col = mapping.get("to_removable")
    if not user_col or not date_col:
        return pd.DataFrame(columns=["user", "day", "files_to_removable"])

    wanted = {user_col, date_col}
    if removable_col:
        wanted.add(removable_col)

    count_parts: list[pd.DataFrame] = []
    for chunk in _open_reader(path, wanted, chunk_size):
        if not {user_col, date_col}.issubset(chunk.columns):
            continue
        parsed = _parse_datetimes(chunk[date_col])
        frame = pd.DataFrame(
            {
                "user": _clean_users(chunk[user_col]),
                "day": parsed.dt.normalize(),
            }
        )
        if removable_col and removable_col in chunk.columns:
            flag = chunk[removable_col].astype("string").str.strip().str.lower()
            is_removable = flag.isin({"1", "true", "yes", "y", "removable"})
            frame = frame[is_removable.values]
        frame = frame[frame["user"].notna() & frame["day"].notna()]
        if frame.empty:
            continue
        frame["files_to_removable"] = 1
        count_parts.append(
            frame.groupby(["user", "day"], as_index=False, sort=False)[
                "files_to_removable"
            ].sum()
        )

    return _sum_parts(count_parts, ["files_to_removable"])


def _extract_http(
    path: Path, mapping: dict[str, str], config: Config, chunk_size: int
) -> pd.DataFrame:
    """Compute the web family per user-day from ``http.csv`` (Req 4.3 / 4.5).

    ``job_site_visits`` and ``upload_leak_site_visits`` count requests whose
    domain is in the configured site lists; ``file_upload_count`` counts
    requests whose URL path contains "upload" (the generator marks an upload
    with a ``/upload`` path — this is a documented heuristic); ``distinct_domains``
    counts distinct domains; ``total_web_requests`` counts all requests.
    """
    user_col = mapping.get("user")
    date_col = mapping.get("date")
    url_col = mapping.get("url")
    if not user_col or not date_col or not url_col:
        return pd.DataFrame(
            columns=["user", "day", "job_site_visits", "upload_leak_site_visits",
                     "file_upload_count", "distinct_domains", "total_web_requests"]
        )

    job_sites = {s.lower() for s in config.site_categories.job_sites}
    leak_sites = {s.lower() for s in config.site_categories.upload_leak_sites}

    wanted = {user_col, date_col, url_col}
    count_parts: list[pd.DataFrame] = []
    domain_parts: list[pd.DataFrame] = []

    for chunk in _open_reader(path, wanted, chunk_size):
        if not wanted.issubset(chunk.columns):
            continue
        parsed = _parse_datetimes(chunk[date_col])
        user = _clean_users(chunk[user_col])
        day = parsed.dt.normalize()
        domain, pathpart = _split_url(chunk[url_col])
        frame = pd.DataFrame(
            {
                "user": user,
                "day": day,
                "domain": domain.values,
                "path": pathpart.values,
            }
        )
        frame = frame[frame["user"].notna() & frame["day"].notna()]
        if frame.empty:
            continue
        agg = pd.DataFrame(
            {
                "user": frame["user"].values,
                "day": frame["day"].values,
                "job_site_visits": frame["domain"].isin(job_sites).astype("int64").values,
                "upload_leak_site_visits": frame["domain"].isin(leak_sites).astype("int64").values,
                "file_upload_count": frame["path"].str.contains("upload", case=False, na=False).astype("int64").values,
                "total_web_requests": 1,
            }
        )
        count_parts.append(
            agg.groupby(["user", "day"], as_index=False, sort=False)[
                ["job_site_visits", "upload_leak_site_visits",
                 "file_upload_count", "total_web_requests"]
            ].sum()
        )
        domains = frame.loc[frame["domain"].str.len() > 0, ["user", "day", "domain"]]
        if not domains.empty:
            domain_parts.append(domains.drop_duplicates())

    counts = _sum_parts(
        count_parts,
        ["job_site_visits", "upload_leak_site_visits", "file_upload_count",
         "total_web_requests"],
    )
    domains = _distinct_count(domain_parts, "domain", "distinct_domains")
    if counts.empty and domains.empty:
        return pd.DataFrame(
            columns=["user", "day", "job_site_visits", "upload_leak_site_visits",
                     "file_upload_count", "distinct_domains", "total_web_requests"]
        )
    return counts.merge(domains, on=["user", "day"], how="outer")


def _extract_email(
    path: Path,
    mapping: dict[str, str],
    config: Config,
    chunk_size: int,
    internal_domain: str,
) -> pd.DataFrame:
    """Compute the email family per user-day from ``email.csv`` (Req 4.4).

    ``emails_sent`` counts rows (one per sent email); ``email_attachments``
    sums the per-email attachment count; ``external_recipients`` counts
    recipients in the ";"-separated "to" field whose domain is **not** the
    internal organisation domain.
    """
    user_col = mapping.get("user")
    date_col = mapping.get("date")
    to_col = mapping.get("to")
    attachments_col = mapping.get("attachments")
    if not user_col or not date_col:
        return pd.DataFrame(
            columns=["user", "day", "emails_sent", "external_recipients",
                     "email_attachments"]
        )

    internal = internal_domain.lower()
    wanted = {user_col, date_col}
    if to_col:
        wanted.add(to_col)
    if attachments_col:
        wanted.add(attachments_col)

    parts: list[pd.DataFrame] = []
    for chunk in _open_reader(path, wanted, chunk_size):
        if not {user_col, date_col}.issubset(chunk.columns):
            continue
        parsed = _parse_datetimes(chunk[date_col])
        frame = pd.DataFrame(
            {
                "user": _clean_users(chunk[user_col]),
                "day": parsed.dt.normalize(),
            }
        )
        if attachments_col and attachments_col in chunk.columns:
            frame["email_attachments"] = (
                pd.to_numeric(chunk[attachments_col], errors="coerce").fillna(0)
            )
        else:
            frame["email_attachments"] = 0
        frame["emails_sent"] = 1
        if to_col and to_col in chunk.columns:
            frame["to"] = chunk[to_col].astype("string").fillna("")
        else:
            frame["to"] = ""
        frame = frame[frame["user"].notna() & frame["day"].notna()]
        if frame.empty:
            continue

        # Per-email counts (emails_sent, attachments) aggregate by (user, day).
        base = frame.groupby(["user", "day"], as_index=False, sort=False)[
            ["emails_sent", "email_attachments"]
        ].sum()

        # External recipients: explode the ";"-separated "to" field, extract
        # each recipient's domain, and count those that are not internal.
        recips = frame[["user", "day", "to"]].copy()
        recips["to"] = recips["to"].str.split(";")
        recips = recips.explode("to")
        recips["to"] = recips["to"].astype("string").str.strip()
        recips = recips[recips["to"].notna() & (recips["to"].str.len() > 0)]
        if recips.empty:
            ext = pd.DataFrame(columns=["user", "day", "external_recipients"])
        else:
            recips["domain"] = recips["to"].str.rsplit("@", n=1).str[-1].str.lower()
            external = recips[recips["domain"] != internal]
            ext = (
                external.groupby(["user", "day"], as_index=False, sort=False)
                .size()
                .rename(columns={"size": "external_recipients"})
            )

        merged = base.merge(ext, on=["user", "day"], how="left")
        merged["external_recipients"] = (
            pd.to_numeric(merged["external_recipients"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
        parts.append(merged)

    return _sum_parts(
        parts, ["emails_sent", "external_recipients", "email_attachments"]
    )


# ---------------------------------------------------------------------------
# Roles and finalisation.
# ---------------------------------------------------------------------------


def _load_roles(raw_dir: Path) -> pd.DataFrame:
    """Load the user -> role lookup from ``roles.csv`` if it is present.

    Returns a two-column frame (``user``, ``role``). A missing file yields an
    empty frame, so roles simply come through as missing rather than causing
    extraction to fail.
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


def _merge_sources(per_source: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join every source's per-user-day features into one table.

    A user-day appears if it had at least one event in any source. Missing
    features (a day with logons but no emails, say) are left as NaN here and
    filled with 0 in :func:`_finalise`.
    """
    frames = [f for f in per_source if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["user", "day", *FEATURE_COLUMNS])
    merged: Optional[pd.DataFrame] = None
    for frame in frames:
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["user", "day"], how="outer")
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
    """Tidy types, fill feature gaps with 0, and order rows and columns.

    Guarantees the output invariants: exactly one row per ``(user, day)``,
    integer feature counts with no missing values, a ``string`` user and role,
    a ``datetime`` day, and the stable column order from design.md.
    """
    if user_day.empty:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty["user"] = empty["user"].astype("string")
        empty["day"] = pd.to_datetime(empty["day"])
        empty["role"] = empty["role"].astype("string")
        for col in FEATURE_COLUMNS:
            empty[col] = empty[col].astype("int64")
        return empty

    user_day = user_day.copy()
    user_day["user"] = user_day["user"].astype("string")
    user_day["day"] = pd.to_datetime(user_day["day"])
    if "role" in user_day.columns:
        user_day["role"] = user_day["role"].astype("string")
    else:
        user_day["role"] = pd.NA

    for col in FEATURE_COLUMNS:
        if col not in user_day.columns:
            user_day[col] = 0
        user_day[col] = (
            pd.to_numeric(user_day[col], errors="coerce").fillna(0).astype("int64")
        )

    user_day = (
        user_day.drop_duplicates(subset=["user", "day"])
        .sort_values(["user", "day"])
        .reset_index(drop=True)
    )
    return user_day[OUTPUT_COLUMNS]


def _write_user_day(user_day: pd.DataFrame, processed_dir: Path) -> Path:
    """Write the full-schema user-day table to ``user_day.parquet``."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / USER_DAY_FILENAME
    user_day.to_parquet(out_path, engine="pyarrow", index=False)
    return out_path


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def extract_features(
    config: Config,
    users: Optional[Iterable[str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    internal_domain: Optional[str] = None,
) -> pd.DataFrame:
    """Build the full per-user-day feature table and write ``user_day.parquet``.

    Reads each raw event CSV in chunks, computes the logon, USB/removable-media,
    web, and email feature families for every ``(user, calendar_day)``, attaches
    each user's role, and writes the full-schema table to
    ``<processed_dir>/user_day.parquet`` (superseding ingestion's coarse
    artifact). After-hours, weekend, and site-category classifications all come
    from ``config`` — nothing is hard-coded.

    Args:
        config: The loaded :class:`~src.ueba.config.Config`. Supplies the raw
            and processed directories, the column map, the working-hours window,
            weekend days, and the site-category lists.
        users: Optional iterable of user ids to restrict the output to (the same
            subset hook ingestion exposes). ``None`` keeps every user found.
        chunk_size: Rows read per chunk; the default keeps any single chunk far
            smaller than a whole raw file.
        internal_domain: Organisation-internal mail domain used to decide which
            recipients are external. Defaults to an ``internal_domain`` attribute
            on ``config`` if present, otherwise ``"dtaa.com"``.

    Returns:
        A DataFrame with one row per ``(user, day)`` carrying ``role`` and all
        feature columns, also written to ``user_day.parquet``.
    """
    raw_dir = Path(config.raw_data_dir)
    processed_dir = Path(config.processed_dir)

    if internal_domain is None:
        internal_domain = getattr(config, "internal_domain", None) or DEFAULT_INTERNAL_DOMAIN

    per_source: list[pd.DataFrame] = []

    # logon.csv -> logon family
    logon_path = raw_dir / "logon.csv"
    if logon_path.is_file():
        per_source.append(
            _extract_logon(logon_path, config.column_map.get("logon", {}), config, chunk_size)
        )

    # device.csv -> USB family
    device_path = raw_dir / "device.csv"
    if device_path.is_file():
        per_source.append(
            _extract_device(device_path, config.column_map.get("device", {}), config, chunk_size)
        )

    # file.csv -> files_to_removable (usually absent on synthetic data)
    file_path = raw_dir / "file.csv"
    if file_path.is_file() and config.column_map.get("file"):
        per_source.append(
            _extract_file(file_path, config.column_map.get("file", {}), config, chunk_size)
        )

    # http.csv -> web family
    http_path = raw_dir / "http.csv"
    if http_path.is_file():
        per_source.append(
            _extract_http(http_path, config.column_map.get("http", {}), config, chunk_size)
        )

    # email.csv -> email family
    email_path = raw_dir / "email.csv"
    if email_path.is_file():
        per_source.append(
            _extract_email(
                email_path, config.column_map.get("email", {}), config,
                chunk_size, internal_domain,
            )
        )

    user_day = _merge_sources(per_source)
    user_day = _attach_roles(user_day, _load_roles(raw_dir))

    if users is not None:
        keep = {str(u) for u in users}
        if not user_day.empty:
            user_day = user_day[user_day["user"].isin(keep)].reset_index(drop=True)

    user_day = _finalise(user_day)
    _write_user_day(user_day, processed_dir)
    return user_day


if __name__ == "__main__":  # pragma: no cover - manual convenience entry point
    from ..config import load_config

    cfg = load_config()
    result = extract_features(cfg)
    print(
        f"[features] Wrote {len(result)} user-day rows to "
        f"{Path(cfg.processed_dir) / USER_DAY_FILENAME}"
    )
