"""Task 2.8 helper — run the real chunked ingestion and verify its invariants.

This is a throwaway verification runner for the optional real-dataset task. It
loads the committed config (which for this run points at the extracted CERT
r4.2 CSVs), samples this process's resident memory on a background thread while
``ingest`` streams the ~15 GB of logs, and then checks the resulting
``user_day.parquet`` against the ingestion acceptance criteria:

* one row per (user, calendar_day) with no duplicates,
* no null/blank user ids,
* the covered date range, user/row counts, and per-source event totals.

It reads the raw directory and column map from config only (nothing is
hard-coded) and never touches labels or evaluation.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

# Ensure the repo root is importable when run as `python scripts/run_real_ingest.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import psutil

from src.ueba.config import load_config
from src.ueba.ingestion.pipeline import ingest, USER_DAY_FILENAME


class MemorySampler(threading.Thread):
    """Poll this process's RSS on an interval and remember the peak."""

    def __init__(self, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self._interval = interval
        self._stop = threading.Event()
        self.peak_bytes = 0
        self._proc = psutil.Process()

    def run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self.peak_bytes:
                self.peak_bytes = rss
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def main() -> int:
    cfg = load_config()
    print(f"[run] raw_data_dir   = {cfg.raw_data_dir}")
    print(f"[run] processed_dir  = {cfg.processed_dir}")
    print(f"[run] column_map     = {cfg.column_map}")

    sampler = MemorySampler(interval=0.5)
    start_rss = psutil.Process().memory_info().rss
    sampler.start()

    t0 = time.time()
    user_day = ingest(cfg)  # full population, chunked; no subset, no labels
    elapsed = time.time() - t0

    sampler.stop()
    sampler.join(timeout=2)

    print("\n=== ingestion complete ===")
    print(f"  elapsed            : {elapsed:.1f} s")
    print(f"  RSS at start       : {_mb(start_rss)}")
    print(f"  peak RSS observed  : {_mb(sampler.peak_bytes)}")
    print(f"  peak growth        : {_mb(sampler.peak_bytes - start_rss)}")

    # Re-load the persisted artifact to verify what actually hit disk.
    out_path = Path(cfg.processed_dir) / USER_DAY_FILENAME
    on_disk = pd.read_parquet(out_path)

    print("\n=== user_day.parquet verification ===")
    print(f"  path               : {out_path}")
    print(f"  rows (returned)    : {len(user_day):,}")
    print(f"  rows (on disk)     : {len(on_disk):,}")
    print(f"  distinct users     : {on_disk['user'].nunique():,}")
    print(f"  columns            : {list(on_disk.columns)}")

    # Invariant: no null/blank user ids.
    null_users = on_disk["user"].isna().sum()
    blank_users = (on_disk["user"].astype("string").str.strip() == "").sum()
    print(f"  null user ids      : {null_users}")
    print(f"  blank user ids     : {blank_users}")

    # Invariant: exactly one row per (user, day).
    dup = on_disk.duplicated(subset=["user", "day"]).sum()
    print(f"  duplicate (user,day): {dup}")

    # Date range covered.
    day_min = pd.to_datetime(on_disk["day"]).min()
    day_max = pd.to_datetime(on_disk["day"]).max()
    print(f"  day range          : {day_min.date()} -> {day_max.date()}")

    # Per-source event totals (sanity of chunk summation).
    for col in ["logon_events", "device_events", "http_events", "email_events", "total_events"]:
        if col in on_disk.columns:
            print(f"  sum {col:<14}: {int(on_disk[col].sum()):,}")

    # Roles come through null when roles.csv is absent (expected for r4.2).
    if "role" in on_disk.columns:
        non_null_roles = on_disk["role"].notna().sum()
        print(f"  non-null roles     : {non_null_roles:,}")

    ok = (null_users == 0 and blank_users == 0 and dup == 0 and len(on_disk) > 0)
    print(f"\n[run] invariants satisfied: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
