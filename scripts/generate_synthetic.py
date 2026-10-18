"""Synthetic_Generator — a stand-in data factory for the UEBA pipeline.

The real CMU/SEI CERT release 4.2 dataset is a 20 GB download. This module
makes small, realistic event logs in the *same shape* as that dataset so the
whole pipeline — ingestion, feature extraction, scoring, and the dashboard —
can run, be tested, and be demoed without the raw files.

It produces four event CSVs matching the CERT 4.2 per-file schema:

* ``logon.csv``  — workstation Logon / Logoff events
* ``device.csv`` — removable-media Connect / Disconnect events
* ``http.csv``   — web requests (a mix of ordinary, job-hunting, and
  upload/leak sites, so the feature layer can categorise them)
* ``email.csv``  — sent email with recipients, attachment counts, and sizes
  (some recipients are external)

It also writes two helper files the real dataset ships separately:

* ``roles.csv``  — one role per user, so peer baselines have groups to compare
  against
* ``labels.csv`` — ground-truth insider user-days (the positive labels). This
  file is written for *evaluation only*; nothing in training reads it, and the
  generator is never required to train the model.

Normal days are generated with genuine day-to-day variation (occasional
after-hours logons, occasional USB use) so each user's baseline has non-zero
spread. :func:`plant_anomaly` then injects a deliberate spike (for example a
USB burst) on a chosen user-day. Because the spike is many times larger than
the baseline's natural variation, the statistical scorer later sees a z-score
well above 4 — large enough to prove the detector catches it.

Run it from the command line, for example::

    python scripts/generate_synthetic.py --out-dir data/raw --n-users 20 \\
        --n-days 120 --seed 42 --plant USER0005:100:usb_burst

Site lists and the working-hours window are read from ``config.yaml`` when it
is available so planted anomalies line up with the feature definitions; if the
config cannot be read the generator falls back to sensible built-in defaults,
keeping it fully self-contained.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema definitions — column order matches the CERT release 4.2 CSV files.
# ---------------------------------------------------------------------------

LOGON_COLUMNS = ["id", "date", "user", "pc", "activity"]
DEVICE_COLUMNS = ["id", "date", "user", "pc", "activity"]
HTTP_COLUMNS = ["id", "date", "user", "pc", "url"]
EMAIL_COLUMNS = [
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
]

# CERT-style timestamp, e.g. "01/04/2010 08:37:12".
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
# Plain calendar day used for the ground-truth labels file.
DAY_FORMAT = "%Y-%m-%d"

# Internal mail domain; anything else counts as an external recipient.
INTERNAL_DOMAIN = "dtaa.com"
EXTERNAL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]

# Ordinary web destinations used to pad out normal browsing.
COMMON_SITES = [
    "google.com",
    "wikipedia.org",
    "bbc.co.uk",
    "nytimes.com",
    "stackoverflow.com",
    "github.com",
    "amazon.com",
    "reddit.com",
]

# Job roles assigned round-robin so peer baselines have groups to compare.
ROLES = [
    "ITAdmin",
    "Salesman",
    "Engineer",
    "Manager",
    "Analyst",
    "HR",
    "Finance",
    "Support",
]

# Fallback config values if config.yaml cannot be read (kept self-contained).
_DEFAULT_JOB_SITES = ["job-hunt.org", "careerbuilder.com"]
_DEFAULT_UPLOAD_LEAK_SITES = ["wikileaks.org", "dropbox.com"]
_DEFAULT_START_HOUR = 8
_DEFAULT_END_HOUR = 18
_DEFAULT_WEEKEND_DAYS = (5, 6)

# Default calendar start: a Monday, so weekday/weekend logic is intuitive.
_DEFAULT_START_DATE = date(2010, 1, 4)


@dataclass
class _GeneratorSettings:
    """Config-derived knobs the generator honours when categorising events."""

    job_sites: tuple[str, ...] = tuple(_DEFAULT_JOB_SITES)
    upload_leak_sites: tuple[str, ...] = tuple(_DEFAULT_UPLOAD_LEAK_SITES)
    start_hour: int = _DEFAULT_START_HOUR
    end_hour: int = _DEFAULT_END_HOUR
    weekend_days: tuple[int, ...] = _DEFAULT_WEEKEND_DAYS


def _load_settings(config_path: str = "config.yaml") -> _GeneratorSettings:
    """Read site lists and the working-hours window from ``config.yaml``.

    Prefers the typed loader in :mod:`src.ueba.config` when it is importable,
    then falls back to reading the YAML directly, and finally to built-in
    defaults. Any failure degrades silently to defaults so the generator never
    depends on config being present.
    """
    # First choice: the project's typed config loader.
    try:
        from src.ueba.config import load_config  # type: ignore

        cfg = load_config(config_path)
        return _GeneratorSettings(
            job_sites=tuple(cfg.site_categories.job_sites),
            upload_leak_sites=tuple(cfg.site_categories.upload_leak_sites),
            start_hour=cfg.working_hours.start_hour,
            end_hour=cfg.working_hours.end_hour,
            weekend_days=tuple(cfg.weekend_days),
        )
    except Exception:
        pass

    # Second choice: read the raw YAML ourselves.
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        site = data.get("site_categories") or {}
        wh = data.get("working_hours") or {}
        weekend = data.get("weekend_days") or list(_DEFAULT_WEEKEND_DAYS)
        return _GeneratorSettings(
            job_sites=tuple(site.get("job_sites") or _DEFAULT_JOB_SITES),
            upload_leak_sites=tuple(
                site.get("upload_leak_sites") or _DEFAULT_UPLOAD_LEAK_SITES
            ),
            start_hour=int(wh.get("start_hour", _DEFAULT_START_HOUR)),
            end_hour=int(wh.get("end_hour", _DEFAULT_END_HOUR)),
            weekend_days=tuple(int(d) for d in weekend),
        )
    except Exception:
        # Final fallback: built-in defaults.
        return _GeneratorSettings()


class SyntheticGenerator:
    """Builds CERT 4.2-shaped event logs and can plant anomalies.

    Typical use::

        gen = SyntheticGenerator()
        gen.generate("data/raw", n_users=20, n_days=120, seed=42)
        gen.plant_anomaly("USER0005", 100, "usb_burst")

    ``generate`` lays down a full set of normal events and writes every CSV.
    ``plant_anomaly`` then injects a spike for one user on one day and rewrites
    the affected files (and the labels file), so the planted day is both in the
    data and marked as ground-truth insider activity.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        self._settings = _load_settings(config_path)
        self._rng: np.random.Generator = np.random.default_rng(0)
        self._id_counter = 0

        # Populated by generate().
        self._out_dir: Optional[str] = None
        self._users: list[str] = []
        self._roles: dict[str, str] = {}
        self._user_pcs: dict[str, list[str]] = {}
        self._start_date: date = _DEFAULT_START_DATE
        self._n_days = 0

        # Event buffers: file type -> list of row dicts.
        self._rows: dict[str, list[dict]] = {
            "logon": [],
            "device": [],
            "http": [],
            "email": [],
        }
        # Ground-truth insider user-days: set of (user, day).
        self._labels: set[tuple[str, date]] = set()

    # -- public API ---------------------------------------------------------

    def generate(
        self, out_dir: str, n_users: int, n_days: int, seed: int
    ) -> None:
        """Generate normal events for every user-day and write all CSVs.

        Args:
            out_dir: Directory to write the CSVs into (created if absent).
            n_users: Number of distinct users to simulate.
            n_days: Number of consecutive calendar days to span.
            seed: Seed for reproducible output.
        """
        if n_users <= 0:
            raise ValueError("n_users must be positive")
        if n_days <= 0:
            raise ValueError("n_days must be positive")

        self._rng = np.random.default_rng(seed)
        self._id_counter = 0
        self._out_dir = out_dir
        self._n_days = n_days
        self._start_date = _DEFAULT_START_DATE

        # Reset buffers in case generate() is called more than once.
        self._rows = {"logon": [], "device": [], "http": [], "email": []}
        self._labels = set()

        self._users = [f"USER{idx:04d}" for idx in range(1, n_users + 1)]
        self._roles = {
            user: ROLES[idx % len(ROLES)] for idx, user in enumerate(self._users)
        }
        # Each user has one or two "home" machines they normally use.
        self._user_pcs = {
            user: [f"PC-{idx:04d}", f"PC-{idx:04d}B"]
            for idx, user in enumerate(self._users, start=1)
        }

        days = [self._start_date + timedelta(days=d) for d in range(n_days)]
        for user in self._users:
            for day in days:
                self._generate_user_day(user, day)

        self._write_all()

    def plant_anomaly(
        self, user_id: str, day: Union[int, str, date, datetime], kind: str = "usb_burst"
    ) -> None:
        """Inject a measurable spike for ``user_id`` on ``day`` and rewrite CSVs.

        Args:
            user_id: The user to make anomalous. Must exist in the generated set.
            day: The target day as a ``date``/``datetime``, a ``YYYY-MM-DD``
                string, or an integer offset from the generated start date.
            kind: The anomaly flavour. One of ``usb_burst``,
                ``after_hours_logons``, ``web_upload``, ``job_hunt``, or
                ``email_exfil``.

        The spike is large relative to the user's natural baseline variation,
        so downstream scoring sees a z-score well above 4. The affected
        user-day is also recorded as a ground-truth insider label.
        """
        if self._out_dir is None:
            raise RuntimeError("Call generate() before plant_anomaly().")
        if user_id not in self._roles:
            raise ValueError(
                f"Unknown user '{user_id}'. Generate users before planting."
            )

        target = self._normalize_day(day)
        kind = kind.lower()

        if kind == "usb_burst":
            self._plant_usb_burst(user_id, target)
        elif kind == "after_hours_logons":
            self._plant_after_hours_logons(user_id, target)
        elif kind == "web_upload":
            self._plant_web_upload(user_id, target)
        elif kind == "job_hunt":
            self._plant_job_hunt(user_id, target)
        elif kind == "email_exfil":
            self._plant_email_exfil(user_id, target)
        else:
            raise ValueError(
                f"Unknown anomaly kind '{kind}'. Valid kinds: usb_burst, "
                "after_hours_logons, web_upload, job_hunt, email_exfil."
            )

        self._labels.add((user_id, target))
        # Rewrite everything so the planted events and label land on disk.
        self._write_all()

    # -- normal-day generation ---------------------------------------------

    def _generate_user_day(self, user: str, day: date) -> None:
        """Lay down a day's worth of ordinary activity for one user."""
        is_weekend = day.weekday() in self._settings.weekend_days

        # On weekends most users are idle; only a few show up.
        if is_weekend and self._rng.random() > 0.15:
            return

        self._generate_logons(user, day, is_weekend)
        self._generate_device(user, day, is_weekend)
        self._generate_http(user, day, is_weekend)
        self._generate_email(user, day, is_weekend)

    def _generate_logons(self, user: str, day: date, is_weekend: bool) -> None:
        """One to three logon/logoff sessions, occasionally after-hours."""
        pc = self._pick_pc(user)
        n_sessions = 1 if is_weekend else int(self._rng.integers(1, 4))
        for _ in range(n_sessions):
            # ~10% of sessions on a normal day start outside working hours.
            after_hours = self._rng.random() < 0.10
            if after_hours:
                start_hour = int(self._rng.choice([6, 7, 19, 20, 21, 22]))
            else:
                start_hour = int(
                    self._rng.integers(
                        self._settings.start_hour, self._settings.end_hour - 1
                    )
                )
            logon_ts = self._timestamp(day, start_hour)
            # A session lasts a few hours.
            logoff_hour = min(start_hour + int(self._rng.integers(1, 5)), 23)
            logoff_ts = self._timestamp(day, logoff_hour)
            self._add("logon", date=logon_ts, user=user, pc=pc, activity="Logon")
            self._add("logon", date=logoff_ts, user=user, pc=pc, activity="Logoff")

    def _generate_device(self, user: str, day: date, is_weekend: bool) -> None:
        """Occasional USB use, giving the baseline a small non-zero spread."""
        p_use = 0.05 if is_weekend else 0.20
        if self._rng.random() >= p_use:
            return
        n_connects = int(self._rng.integers(1, 3))  # 1 or 2 connects
        pc = self._pick_pc(user)
        for _ in range(n_connects):
            hour = int(
                self._rng.integers(
                    self._settings.start_hour, self._settings.end_hour
                )
            )
            connect_ts = self._timestamp(day, hour)
            disconnect_ts = self._timestamp(day, min(hour + 1, 23))
            self._add("device", date=connect_ts, user=user, pc=pc, activity="Connect")
            self._add(
                "device", date=disconnect_ts, user=user, pc=pc, activity="Disconnect"
            )

    def _generate_http(self, user: str, day: date, is_weekend: bool) -> None:
        """Normal browsing, with a small chance of a job-site visit."""
        mean = 8 if is_weekend else 30
        n_requests = int(self._rng.poisson(mean))
        pc = self._pick_pc(user)
        for _ in range(n_requests):
            # ~3% of normal requests hit a job site (low-level background).
            if self._rng.random() < 0.03:
                domain = str(self._rng.choice(self._settings.job_sites))
            else:
                domain = str(self._rng.choice(COMMON_SITES))
            hour = int(self._rng.integers(self._settings.start_hour, self._settings.end_hour))
            ts = self._timestamp(day, hour)
            self._add("http", date=ts, user=user, pc=pc, url=self._url(domain))

    def _generate_email(self, user: str, day: date, is_weekend: bool) -> None:
        """A handful of emails, some external, some with attachments."""
        mean = 2 if is_weekend else 6
        n_emails = int(self._rng.poisson(mean))
        pc = self._pick_pc(user)
        for _ in range(n_emails):
            external = self._rng.random() < 0.25
            recipients = self._recipients(external=external, count=int(self._rng.integers(1, 4)))
            attachments = int(self._rng.integers(0, 3))
            size = int(self._rng.integers(2_000, 50_000))
            hour = int(self._rng.integers(self._settings.start_hour, self._settings.end_hour))
            ts = self._timestamp(day, hour)
            self._add(
                "email",
                date=ts,
                user=user,
                pc=pc,
                to=recipients,
                cc="",
                bcc="",
                **{"from": self._address(user)},
                size=size,
                attachments=attachments,
                content="",
            )

    # -- anomaly planters ---------------------------------------------------

    def _plant_usb_burst(self, user: str, day: date) -> None:
        """A flood of USB Connect/Disconnect events, some after-hours."""
        pc = self._pick_pc(user)
        for i in range(30):
            # Spread across the day, including evenings, to spike after-hours USB too.
            hour = int(self._rng.integers(0, 24))
            connect_ts = self._timestamp(day, hour)
            disconnect_ts = self._timestamp(day, min(hour + 1, 23))
            self._add("device", date=connect_ts, user=user, pc=pc, activity="Connect")
            self._add(
                "device", date=disconnect_ts, user=user, pc=pc, activity="Disconnect"
            )

    def _plant_after_hours_logons(self, user: str, day: date) -> None:
        """Many logon/logoff pairs late at night."""
        pc = self._pick_pc(user)
        for _ in range(12):
            start_hour = int(self._rng.choice([0, 1, 2, 3, 22, 23]))
            logon_ts = self._timestamp(day, start_hour)
            logoff_ts = self._timestamp(day, min(start_hour + 1, 23))
            self._add("logon", date=logon_ts, user=user, pc=pc, activity="Logon")
            self._add("logon", date=logoff_ts, user=user, pc=pc, activity="Logoff")

    def _plant_web_upload(self, user: str, day: date) -> None:
        """A burst of visits to upload/leak sites."""
        pc = self._pick_pc(user)
        for _ in range(40):
            domain = str(self._rng.choice(self._settings.upload_leak_sites))
            hour = int(self._rng.integers(0, 24))
            ts = self._timestamp(day, hour)
            self._add("http", date=ts, user=user, pc=pc, url=self._url(domain, upload=True))

    def _plant_job_hunt(self, user: str, day: date) -> None:
        """A burst of job-site visits."""
        pc = self._pick_pc(user)
        for _ in range(40):
            domain = str(self._rng.choice(self._settings.job_sites))
            hour = int(self._rng.integers(0, 24))
            ts = self._timestamp(day, hour)
            self._add("http", date=ts, user=user, pc=pc, url=self._url(domain))

    def _plant_email_exfil(self, user: str, day: date) -> None:
        """Emails to many external recipients with large attachments."""
        pc = self._pick_pc(user)
        for _ in range(15):
            recipients = self._recipients(external=True, count=int(self._rng.integers(3, 8)))
            self._add(
                "email",
                date=self._timestamp(day, int(self._rng.integers(0, 24))),
                user=user,
                pc=pc,
                to=recipients,
                cc="",
                bcc="",
                **{"from": self._address(user)},
                size=int(self._rng.integers(500_000, 5_000_000)),
                attachments=int(self._rng.integers(3, 8)),
                content="",
            )

    # -- writing ------------------------------------------------------------

    def _write_all(self) -> None:
        """Write every event CSV plus roles.csv and labels.csv to out_dir."""
        assert self._out_dir is not None
        os.makedirs(self._out_dir, exist_ok=True)

        self._write_csv("logon.csv", self._rows["logon"], LOGON_COLUMNS)
        self._write_csv("device.csv", self._rows["device"], DEVICE_COLUMNS)
        self._write_csv("http.csv", self._rows["http"], HTTP_COLUMNS)
        self._write_csv("email.csv", self._rows["email"], EMAIL_COLUMNS)
        self._write_roles()
        self._write_labels()

    def _write_csv(self, name: str, rows: list[dict], columns: list[str]) -> None:
        """Write one event file, preserving the CERT column order."""
        assert self._out_dir is not None
        frame = pd.DataFrame(rows, columns=columns)
        frame.to_csv(os.path.join(self._out_dir, name), index=False)

    def _write_roles(self) -> None:
        """Write the per-user role assignment used for peer baselines."""
        assert self._out_dir is not None
        frame = pd.DataFrame(
            [{"user": user, "role": role} for user, role in sorted(self._roles.items())],
            columns=["user", "role"],
        )
        frame.to_csv(os.path.join(self._out_dir, "roles.csv"), index=False)

    def _write_labels(self) -> None:
        """Write ground-truth insider user-days (evaluation only)."""
        assert self._out_dir is not None
        rows = [
            {"user": user, "day": day.strftime(DAY_FORMAT), "insider": 1}
            for user, day in sorted(self._labels, key=lambda item: (item[0], item[1]))
        ]
        frame = pd.DataFrame(rows, columns=["user", "day", "insider"])
        frame.to_csv(os.path.join(self._out_dir, "labels.csv"), index=False)

    # -- small helpers ------------------------------------------------------

    def _add(self, file_type: str, **fields) -> None:
        """Append one row (with a fresh unique id) to an event buffer."""
        self._id_counter += 1
        row = {"id": f"{{R{self._id_counter:07d}}}"}
        row.update(fields)
        self._rows[file_type].append(row)

    def _timestamp(self, day: date, hour: int) -> str:
        """Format a CERT-style timestamp for a day at a given hour."""
        minute = int(self._rng.integers(0, 60))
        second = int(self._rng.integers(0, 60))
        ts = datetime(day.year, day.month, day.day, hour, minute, second)
        return ts.strftime(DATE_FORMAT)

    def _pick_pc(self, user: str) -> str:
        """Return one of the user's home machines."""
        return str(self._rng.choice(self._user_pcs[user]))

    def _url(self, domain: str, upload: bool = False) -> str:
        """Build a plausible URL for a domain."""
        path = "upload" if upload else str(self._rng.choice(["", "index", "search", "page"]))
        return f"http://{domain}/{path}".rstrip("/")

    def _address(self, user: str) -> str:
        """Internal email address for a user."""
        return f"{user.lower()}@{INTERNAL_DOMAIN}"

    def _recipients(self, external: bool, count: int) -> str:
        """Semicolon-joined recipient list, optionally including externals."""
        recips: list[str] = []
        for _ in range(max(count, 1)):
            if external and self._rng.random() < 0.6:
                domain = str(self._rng.choice(EXTERNAL_DOMAINS))
                recips.append(f"contact{int(self._rng.integers(1, 9999))}@{domain}")
            else:
                peer = str(self._rng.choice(self._users)) if self._users else "USER0001"
                recips.append(f"{peer.lower()}@{INTERNAL_DOMAIN}")
        return ";".join(recips)

    def _normalize_day(self, day: Union[int, str, date, datetime]) -> date:
        """Coerce a day argument into a ``date`` within the generated range."""
        if isinstance(day, bool):  # guard: bool is an int subclass
            raise TypeError("day must be an int offset, date, datetime, or string.")
        if isinstance(day, int):
            if not 0 <= day < self._n_days:
                raise ValueError(
                    f"Day offset {day} is outside the generated range "
                    f"[0, {self._n_days})."
                )
            return self._start_date + timedelta(days=day)
        if isinstance(day, datetime):
            return day.date()
        if isinstance(day, date):
            return day
        if isinstance(day, str):
            return datetime.strptime(day, DAY_FORMAT).date()
        raise TypeError(f"Unsupported day type: {type(day).__name__}")


# ---------------------------------------------------------------------------
# Module-level convenience functions matching the design's interface.
# ---------------------------------------------------------------------------

_GENERATOR: Optional[SyntheticGenerator] = None


def _get_generator() -> SyntheticGenerator:
    """Return the shared module-level generator, creating it on first use."""
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = SyntheticGenerator()
    return _GENERATOR


def generate(out_dir: str, n_users: int, n_days: int, seed: int) -> None:
    """Generate CERT 4.2-shaped CSVs into ``out_dir`` (design interface)."""
    _get_generator().generate(out_dir, n_users, n_days, seed)


def plant_anomaly(
    user_id: str, day: Union[int, str, date, datetime], kind: str = "usb_burst"
) -> None:
    """Plant an anomaly for ``user_id`` on ``day`` (design interface)."""
    _get_generator().plant_anomaly(user_id, day, kind)


# ---------------------------------------------------------------------------
# Command-line entry point.
# ---------------------------------------------------------------------------

def _parse_plant_spec(spec: str) -> tuple[str, Union[int, str], str]:
    """Parse a ``user:day:kind`` CLI plant specification."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid --plant spec '{spec}'. Expected USER:DAY:KIND."
        )
    user, day_raw, kind = parts
    day: Union[int, str]
    try:
        day = int(day_raw)
    except ValueError:
        day = day_raw  # treat as YYYY-MM-DD
    return user, day, kind


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: generate synthetic logs and optionally plant anomalies."""
    parser = argparse.ArgumentParser(
        description="Generate CERT 4.2-shaped synthetic event logs."
    )
    parser.add_argument("--out-dir", default="data/raw", help="Output directory.")
    parser.add_argument("--n-users", type=int, default=20, help="Number of users.")
    parser.add_argument("--n-days", type=int, default=120, help="Number of days.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument(
        "--plant",
        action="append",
        default=[],
        type=_parse_plant_spec,
        help="Plant an anomaly, e.g. USER0005:100:usb_burst. May repeat.",
    )
    args = parser.parse_args(argv)

    gen = SyntheticGenerator(config_path=args.config)
    gen.generate(args.out_dir, args.n_users, args.n_days, args.seed)
    print(
        f"[generate] Wrote synthetic logs for {args.n_users} users over "
        f"{args.n_days} days to {args.out_dir}"
    )
    for user, day, kind in args.plant:
        gen.plant_anomaly(user, day, kind)
        print(f"[generate] Planted '{kind}' anomaly for {user} on day {day}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
