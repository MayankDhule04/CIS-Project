"""Prepare the real CERT r4.2 dataset for evaluation (offline).

This ties the real-data adapters together into a single command. It builds the
two helper files the pipeline needs from the real dataset's own layout:

* ``roles.csv`` — from the monthly LDAP snapshots, via
  :func:`src.ueba.realdata.adapters.build_roles_from_ldap`.
* ``labels.csv`` — from the CERT answers (an ``answers/`` directory or a
  published ``insiders.csv``), via
  :func:`src.ueba.realdata.adapters.build_labels_from_answers`. Built only when
  ``--answers`` is supplied, because CERT ground truth ships separately.

Both files are written into the raw directory so ingestion, feature extraction,
subset selection, and evaluation find them exactly where they already look.

Usage::

    python scripts/prepare_real_data.py
    python scripts/prepare_real_data.py --answers path/to/answers
    python scripts/prepare_real_data.py --answers path/to/insiders.csv

It runs entirely offline and makes no network calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``import src.ueba...`` work from the repo root regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ueba.realdata.adapters import (  # noqa: E402
    LABELS_FILENAME,
    ROLES_FILENAME,
    build_labels_for_release,
    build_labels_from_answers,
    build_roles_from_ldap,
)

# Default location of the extracted real CERT r4.2 CSVs (the nested dir).
_DEFAULT_RAW_DIR = "data/raw/r4.2/r4.2"


def main(argv: "list[str] | None" = None) -> int:
    """Build roles.csv (and optionally labels.csv) into the raw directory."""
    parser = argparse.ArgumentParser(
        description="Build roles.csv and labels.csv for the real CERT r4.2 data."
    )
    parser.add_argument(
        "--raw-dir",
        default=_DEFAULT_RAW_DIR,
        help=f"Directory with the real CERT CSVs (default: {_DEFAULT_RAW_DIR}).",
    )
    parser.add_argument(
        "--ldap-dir",
        default=None,
        help="Directory of monthly LDAP snapshots (default: <raw-dir>/LDAP).",
    )
    parser.add_argument(
        "--answers",
        default=None,
        help="Path to the CERT answers directory or a published insiders.csv. "
        "When omitted, only roles.csv is built (labels ship separately).",
    )
    parser.add_argument(
        "--release",
        default=None,
        help="Restrict label building to a single CERT release's scenario "
        "directories (e.g. 'r4.2'). Required when --answers points at the "
        "multi-release CERT answers/ tree, so other releases' insiders are "
        "excluded. When omitted, --answers is mined as-is (single file or tree).",
    )
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    ldap_dir = Path(args.ldap_dir) if args.ldap_dir else raw_dir / "LDAP"

    if not raw_dir.is_dir():
        print(f"[error] raw dir not found: {raw_dir}", file=sys.stderr)
        return 1

    print("=== prepare real data ===")
    print(f"  raw dir   : {raw_dir}")
    print(f"  ldap dir  : {ldap_dir}")

    # 1) roles.csv from LDAP.
    roles_out = raw_dir / ROLES_FILENAME
    roles = build_roles_from_ldap(ldap_dir, out_path=roles_out)
    print(f"\n[roles] wrote {roles_out}")
    print(f"[roles] users mapped : {len(roles):,}")
    sample = roles.head(5)
    for _, row in sample.iterrows():
        print(f"[roles]   {row['user']} -> {row['role']}")

    # 2) labels.csv from answers (optional).
    if args.answers:
        labels_out = raw_dir / LABELS_FILENAME
        if args.release:
            print(f"\n[labels] release filter : {args.release}")
            labels = build_labels_for_release(
                args.answers, release=args.release, out_path=labels_out
            )
        else:
            labels = build_labels_from_answers(args.answers, out_path=labels_out)
        n_users = labels["user"].nunique() if not labels.empty else 0
        print(f"[labels] wrote {labels_out}")
        print(f"[labels] positive user-days : {len(labels):,}")
        print(f"[labels] distinct insiders  : {n_users:,}")
    else:
        print("\n[labels] skipped (no --answers given).")
        print("[labels] CERT r4.2 ground truth ships separately. Once you have")
        print("[labels] the answers, re-run with --answers <path> to write")
        print(f"[labels] {raw_dir / LABELS_FILENAME}.")

    print("\n=== next steps ===")
    print("  1. Ensure labels.csv exists (re-run with --answers once you have it).")
    print("  2. Point config.yaml paths.raw_data_dir at the real nested dir:")
    print(f"       raw_data_dir: {raw_dir.as_posix()}")
    print("  3. Build risk + report from the real data:")
    print("       python scripts/run_eval.py --rebuild")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
