"""Near-real-time watcher: score new days as they arrive (daily micro-batch).

This polls on an interval and calls the incremental scorer, which freezes the
baselines/model on first run, scores any day not yet in ``risk.parquet``, and
appends the result. The dashboard reads ``risk.parquet``, so it reflects new
days on refresh. This is near-real-time at DAILY cadence - it is not per-event
streaming - and it runs entirely offline.

Usage::

    python scripts/live_watch.py                 # score once and exit
    python scripts/live_watch.py --interval 300  # re-check every 5 minutes
    python scripts/live_watch.py --as-of 2011-05-16

Point it at the real data by setting config.yaml paths.raw_data_dir, or pass a
different --config.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ueba.config import load_config  # noqa: E402
from ueba.live import incremental_update  # noqa: E402


def _run_once(config, as_of) -> int:
    """Run one incremental pass and return the number of new user-day rows."""
    new_rows = incremental_update(config, as_of_day=as_of)
    n = 0 if new_rows is None else len(new_rows)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if n:
        users = new_rows["user"].nunique()
        print(f"[{stamp}] scored {n} new user-days for {users} users")
    else:
        print(f"[{stamp}] no new days to score")
    return n


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Score new days and append to risk.parquet.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--as-of", default=None, help="Only score this day (YYYY-MM-DD).")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Seconds between checks. 0 (default) runs once and exits.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.interval and args.interval > 0:
        print(f"[watch] polling every {args.interval:.0f}s - Ctrl+C to stop")
        try:
            while True:
                _run_once(config, args.as_of)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[watch] stopped")
            return 0
    else:
        _run_once(config, args.as_of)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
