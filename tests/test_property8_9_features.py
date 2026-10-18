"""Property-based tests for the Feature_Extractor.

Feature: ueba-insider-threat-detection, Property 8: Feature counts equal
independently derived truth.

Feature: ueba-insider-threat-detection, Property 9: Classification follows
config.

These tests drive ``src.ueba.features.extractor.extract_features`` against
small, fully controlled raw event CSVs that Hypothesis generates directly
(logon / device / http / email, matching the CERT 4.2 schema). For every
generated example the test writes the CSVs into a per-example
``tempfile.TemporaryDirectory`` created inside the test body — the
function-scoped ``tmp_path`` fixture is deliberately avoided because combining
it with ``@given`` trips a Hypothesis health check — points a
``dataclasses.replace`` copy of the real :class:`Config` at that temp dir, runs
the extractor, and compares every feature column against a truth table computed
independently from the same input events.

Property 8 checks all 15 feature counts for arbitrary event sets. Property 9
varies the working-hours window and the site-category lists across examples and
checks that the after-hours flag and the site-category classification track the
configured values exactly.
"""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from dataclasses import replace
from datetime import datetime

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ueba.config import SiteCategories, WorkingHours, load_config
from ueba.features.extractor import extract_features

# Repository root -> the real config.yaml, loaded once and reused. Only the raw
# and processed directories (and, for Property 9, the working-hours window and
# site lists) are swapped out per example via dataclasses.replace.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")
_BASE_CONFIG = load_config(_CONFIG_PATH)

# CERT-style timestamp written into the event CSVs.
_FMT = "%m/%d/%Y %H:%M:%S"

# The full ordered feature list, mirrored here independently of the extractor's
# own constant so the test genuinely validates the produced columns.
_FEATURES = [
    "logon_count",
    "after_hours_logon_count",
    "weekend_logon_count",
    "distinct_machines",
    "usb_connect_count",
    "after_hours_usb_count",
    "files_to_removable",
    "job_site_visits",
    "upload_leak_site_visits",
    "file_upload_count",
    "distinct_domains",
    "total_web_requests",
    "emails_sent",
    "external_recipients",
    "email_attachments",
]

# Controlled value pools. Domains deliberately include the default config's job
# sites (job-hunt.org, careerbuilder.com) and upload/leak sites (wikileaks.org,
# dropbox.com) plus neutral domains, so site-category counting is exercised.
_USERS = ["USER0001", "USER0002", "USER0003"]
_PCS = ["PC-1", "PC-2", "PC-3"]
_DOMAINS = [
    "job-hunt.org",
    "careerbuilder.com",
    "wikileaks.org",
    "dropbox.com",
    "google.com",
    "intranet.local",
]
# Paths after the first "/"; some contain "upload" (any case) to drive the
# file-upload heuristic, "" means a URL with no path.
_PATHS = ["", "home", "upload", "Upload", "reports/upload", "search", "docs"]
# Recipient addresses mixing the internal org domain (dtaa.com) with externals.
_EMAIL_ADDRS = [
    "alice@dtaa.com",
    "bob@gmail.com",
    "carol@external.org",
    "dave@dtaa.com",
    "eve@partner.net",
]

_INTERNAL_DOMAIN = "dtaa.com"

# Reusable element strategies.
_user = st.sampled_from(_USERS)
_ts = st.datetimes(
    min_value=datetime(2010, 1, 1), max_value=datetime(2010, 12, 31, 23, 59, 59)
)
_pc = st.sampled_from(_PCS)
_logon_row = st.tuples(_user, _ts, _pc, st.sampled_from(["Logon", "Logoff"]))
_device_row = st.tuples(_user, _ts, _pc, st.sampled_from(["Connect", "Disconnect"]))
_http_row = st.tuples(_user, _ts, _pc, st.sampled_from(_DOMAINS), st.sampled_from(_PATHS))
_email_row = st.tuples(
    _user,
    _ts,
    _pc,
    st.lists(st.sampled_from(_EMAIL_ADDRS), min_size=0, max_size=4),
    st.integers(min_value=0, max_value=5),
)


# ---------------------------------------------------------------------------
# CSV writers — emit the CERT 4.2 per-file schema so extract_features reads the
# generated events exactly as it would read real logs.
# ---------------------------------------------------------------------------


def _url(domain: str, path: str) -> str:
    """Build a URL whose parsed domain/path are exactly ``domain``/``path``."""
    return f"http://{domain}/{path}" if path else f"http://{domain}"


def _write_logon(raw_dir: str, rows) -> None:
    pd.DataFrame(
        {
            "id": list(range(len(rows))),
            "date": [r[1].strftime(_FMT) for r in rows],
            "user": [r[0] for r in rows],
            "pc": [r[2] for r in rows],
            "activity": [r[3] for r in rows],
        }
    ).to_csv(os.path.join(raw_dir, "logon.csv"), index=False)


def _write_device(raw_dir: str, rows) -> None:
    pd.DataFrame(
        {
            "id": list(range(len(rows))),
            "date": [r[1].strftime(_FMT) for r in rows],
            "user": [r[0] for r in rows],
            "pc": [r[2] for r in rows],
            "activity": [r[3] for r in rows],
        }
    ).to_csv(os.path.join(raw_dir, "device.csv"), index=False)


def _write_http(raw_dir: str, rows) -> None:
    pd.DataFrame(
        {
            "id": list(range(len(rows))),
            "date": [r[1].strftime(_FMT) for r in rows],
            "user": [r[0] for r in rows],
            "pc": [r[2] for r in rows],
            "url": [_url(r[3], r[4]) for r in rows],
        }
    ).to_csv(os.path.join(raw_dir, "http.csv"), index=False)


def _write_email(raw_dir: str, rows) -> None:
    pd.DataFrame(
        {
            "id": list(range(len(rows))),
            "date": [r[1].strftime(_FMT) for r in rows],
            "user": [r[0] for r in rows],
            "pc": [r[2] for r in rows],
            "to": [";".join(r[3]) for r in rows],
            "cc": ["" for _ in rows],
            "bcc": ["" for _ in rows],
            "from": [f"{r[0]}@{_INTERNAL_DOMAIN}" for r in rows],
            "size": [0 for _ in rows],
            "attachments": [int(r[4]) for r in rows],
            "content": ["" for _ in rows],
        }
    ).to_csv(os.path.join(raw_dir, "email.csv"), index=False)


def _write_all(raw_dir: str, logon, device, http, email) -> None:
    _write_logon(raw_dir, logon)
    _write_device(raw_dir, device)
    _write_http(raw_dir, http)
    _write_email(raw_dir, email)


# ---------------------------------------------------------------------------
# Independent truth computation (does not reuse any extractor internals).
# ---------------------------------------------------------------------------


def _compute_truth(logon, device, http, email, start, end, weekend, job, leak, internal):
    """Derive the expected per-(user, day) feature table straight from events.

    Mirrors the documented feature semantics without touching the extractor:
    a user-day exists if any source contributed it, and each feature is the
    plain count/sum over the events that fed that user-day.
    """

    def is_after_hours(hour: int) -> bool:
        return hour < start or hour >= end

    feat = defaultdict(lambda: {col: 0 for col in _FEATURES})
    machines = defaultdict(set)
    domains = defaultdict(set)
    seen = set()

    # Logon family (Req 4.1). distinct_machines spans every logon.csv row
    # (any activity) with a non-empty pc, matching the extractor.
    for user, ts, pc, activity in logon:
        key = (user, ts.date())
        seen.add(key)
        feat[key]
        if activity.lower() == "logon":
            feat[key]["logon_count"] += 1
            if is_after_hours(ts.hour):
                feat[key]["after_hours_logon_count"] += 1
            if ts.weekday() in weekend:
                feat[key]["weekend_logon_count"] += 1
        if pc and len(str(pc)) > 0:
            machines[key].add(pc)

    # USB / removable-media family (Req 4.2). Only Connect rows create a
    # user-day; files_to_removable stays 0 (no file.csv is written).
    for user, ts, pc, activity in device:
        if activity.lower() != "connect":
            continue
        key = (user, ts.date())
        seen.add(key)
        feat[key]["usb_connect_count"] += 1
        if is_after_hours(ts.hour):
            feat[key]["after_hours_usb_count"] += 1

    # Web family (Req 4.3 / 4.5).
    for user, ts, pc, domain, path in http:
        key = (user, ts.date())
        seen.add(key)
        dom = domain.lower()
        feat[key]["total_web_requests"] += 1
        if dom in job:
            feat[key]["job_site_visits"] += 1
        if dom in leak:
            feat[key]["upload_leak_site_visits"] += 1
        if "upload" in path.lower():
            feat[key]["file_upload_count"] += 1
        if dom:
            domains[key].add(dom)

    # Email family (Req 4.4).
    for user, ts, pc, to_list, attachments in email:
        key = (user, ts.date())
        seen.add(key)
        feat[key]["emails_sent"] += 1
        feat[key]["email_attachments"] += int(attachments)
        for addr in to_list:
            stripped = addr.strip()
            if not stripped:
                continue
            recipient_domain = stripped.rsplit("@", 1)[-1].lower()
            if recipient_domain != internal:
                feat[key]["external_recipients"] += 1

    for key in seen:
        feat[key]["distinct_machines"] = len(machines[key])
        feat[key]["distinct_domains"] = len(domains[key])
        feat[key]["files_to_removable"] = 0

    return {key: feat[key] for key in seen}


# ---------------------------------------------------------------------------
# Property 8
# ---------------------------------------------------------------------------


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    logon=st.lists(_logon_row, max_size=12),
    device=st.lists(_device_row, max_size=12),
    http=st.lists(_http_row, max_size=12),
    email=st.lists(_email_row, max_size=12),
)
def test_feature_counts_equal_independent_truth(logon, device, http, email):
    """Property 8: every extracted feature equals the independent count.

    Feature: ueba-insider-threat-detection, Property 8: Feature counts equal
    independently derived truth. Validates Requirements 4.1, 4.2, 4.3, 4.4.

    For an arbitrary set of logon/device/http/email events, the extractor's
    per-(user, day) feature table must match — row set and every one of the 15
    feature columns — the truth table computed directly from those events.
    """
    start = _BASE_CONFIG.working_hours.start_hour
    end = _BASE_CONFIG.working_hours.end_hour
    weekend = set(_BASE_CONFIG.weekend_days)
    job = {s.lower() for s in _BASE_CONFIG.site_categories.job_sites}
    leak = {s.lower() for s in _BASE_CONFIG.site_categories.upload_leak_sites}

    truth = _compute_truth(
        logon, device, http, email, start, end, weekend, job, leak, _INTERNAL_DOMAIN
    )

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = os.path.join(tmp, "raw")
        proc_dir = os.path.join(tmp, "proc")
        os.makedirs(raw_dir)
        _write_all(raw_dir, logon, device, http, email)

        config = replace(_BASE_CONFIG, raw_data_dir=raw_dir, processed_dir=proc_dir)
        result = extract_features(config)

    result_keys = {
        (str(row.user), pd.Timestamp(row.day).date()) for row in result.itertuples()
    }
    assert result_keys == set(truth), (
        f"user-day row set mismatch: extra={result_keys - set(truth)}, "
        f"missing={set(truth) - result_keys}"
    )

    for row in result.itertuples():
        key = (str(row.user), pd.Timestamp(row.day).date())
        expected = truth[key]
        for col in _FEATURES:
            actual = int(getattr(row, col))
            assert actual == expected[col], (
                f"{col} mismatch for {key}: extractor={actual}, truth={expected[col]}"
            )


# ---------------------------------------------------------------------------
# Property 9
# ---------------------------------------------------------------------------


@st.composite
def _classification_cases(draw):
    """Draw a timestamp, a valid working-hours window, a domain, and site lists.

    ``end`` ranges up to 24 (exclusive upper bound) so windows like [8, 24)
    where nothing is after-hours above start are covered too. Job and leak
    lists are arbitrary subsets of the domain pool, so the tested domain may
    fall in neither, either, or both.
    """
    ts = draw(_ts)
    start = draw(st.integers(min_value=0, max_value=23))
    end = draw(st.integers(min_value=start + 1, max_value=24))
    domain = draw(st.sampled_from(_DOMAINS))
    job = draw(st.lists(st.sampled_from(_DOMAINS), max_size=4, unique=True))
    leak = draw(st.lists(st.sampled_from(_DOMAINS), max_size=4, unique=True))
    return ts, start, end, domain, tuple(job), tuple(leak)


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(case=_classification_cases())
def test_classification_follows_config(case):
    """Property 9: after-hours and site classification track the config.

    Feature: ueba-insider-threat-detection, Property 9: Classification follows
    config. Validates Requirements 4.5.

    For any timestamp, any working-hours window, and any site-category lists,
    the after-hours logon/USB flags equal whether the timestamp's hour falls
    outside [start_hour, end_hour), and the job-site and upload/leak-site visit
    counts equal the domain's membership in the configured lists.
    """
    ts, start, end, domain, job, leak = case
    user = "USER0001"
    pc = "PC-1"

    logon = [(user, ts, pc, "Logon")]
    device = [(user, ts, pc, "Connect")]
    http = [(user, ts, pc, domain, "home")]  # neutral path (no "upload")
    email: list = []

    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = os.path.join(tmp, "raw")
        proc_dir = os.path.join(tmp, "proc")
        os.makedirs(raw_dir)
        _write_all(raw_dir, logon, device, http, email)

        config = replace(
            _BASE_CONFIG,
            raw_data_dir=raw_dir,
            processed_dir=proc_dir,
            working_hours=WorkingHours(start_hour=start, end_hour=end),
            site_categories=SiteCategories(job_sites=job, upload_leak_sites=leak),
        )
        result = extract_features(config)

    assert len(result) == 1, f"expected a single user-day, got {len(result)}"
    row = result.iloc[0]

    expected_after_hours = 1 if (ts.hour < start or ts.hour >= end) else 0
    assert int(row["after_hours_logon_count"]) == expected_after_hours
    assert int(row["after_hours_usb_count"]) == expected_after_hours

    dom = domain.lower()
    job_set = {s.lower() for s in job}
    leak_set = {s.lower() for s in leak}
    assert int(row["job_site_visits"]) == (1 if dom in job_set else 0)
    assert int(row["upload_leak_site_visits"]) == (1 if dom in leak_set else 0)
