"""Dashboard example and integration tests (task 6.4).

Covers the two Phase 6 acceptance checks that need the *rendered* Streamlit app
rather than the pure logic layer:

* **Example (Req 8.2):** the date control and the risk-threshold control render.
  We run ``app.py`` headlessly with Streamlit's ``AppTest`` harness and assert a
  ``date_input`` widget and a ``slider`` widget are present (the date range and
  the Daily_Risk threshold, respectively) and that the script raised no
  exception.

* **Integration (Req 8.6):** the dashboard becomes interactive within 10 seconds
  on a Subset-sized risk frame. We build a representative Subset (~310 users over
  a 60-day span — all insiders plus the configured ``normal_sample_size`` of
  normals), then measure the wall-clock time to (a) load ``risk.parquet`` and
  (b) run the app's first render. Both must finish well under 10 seconds.

Design choices worth knowing:

* The dashboard reads one pre-computed artifact, ``<processed_dir>/risk.parquet``
  (see ``src/ueba/scoring/engine.py``). ``app.py`` loads its own config, so these
  tests can't redirect it to a temp dir — the app always looks at the repo's
  configured ``processed_dir``. So the fixture writes a synthetic ``risk.parquet``
  there, **backing up any pre-existing file and restoring it afterwards** so the
  repo is left exactly as it was. Only ``risk.parquet`` is touched; no raw CSVs
  are created.
* The synthetic frame is built directly against the engine's Arrow schema via
  :func:`src.ueba.scoring.engine.write_risk`, not by running the full
  ingestion→scoring pipeline. That keeps the fixture deterministic and fast while
  producing a file the dashboard reads through exactly the same code path as a
  real run.
* ``AppTest`` runs the script in-process. ``tests/conftest.py`` has already put
  the repo root on ``sys.path`` (so ``app.py``'s ``from src.ueba...`` imports
  resolve), and pytest runs from the repo root (so the app's relative
  ``config.yaml`` / ``data/processed`` paths resolve).

These tests do not modify any implementation. If a genuine bug surfaces, the test
will fail loudly rather than paper over it.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ueba.config import load_config
from src.ueba.dashboard.logic import load_risk, rank_users, risk_parquet_path
from src.ueba.scoring.engine import write_risk

# Streamlit's headless app-testing harness (present in modern Streamlit).
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

# The 10-second interactivity budget from Requirement 8.6.
_INTERACTIVE_BUDGET_SECONDS = 10.0

# Calendar start for the synthetic span — a fixed Monday so the data is
# deterministic across runs.
_START_DAY = pd.Timestamp("2010-01-04")

# Days of history per user in the synthetic Subset. Long enough to exercise the
# timeline and alerts-per-day views, short enough to stay fast.
_N_DAYS = 60

# Number of "insider" users given an elevated risk profile. The rest are normals
# sized to the configured subset.normal_sample_size.
_N_INSIDERS = 10


def _build_subset_risk_frame(n_normals: int, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic, Subset-sized risk frame matching risk.parquet.

    Produces one row per (user, day) for ``n_normals`` normal users plus
    :data:`_N_INSIDERS` insiders over :data:`_N_DAYS` consecutive days, carrying
    exactly the columns the Scoring_Engine writes (``user``, ``day``,
    ``self_score``, ``isolation_score``, ``peer_score``, ``daily_risk``,
    ``weekly_risk``, ``reasons``). Insiders get a higher Daily_Risk so the
    ranking, threshold, and drill-down views have meaningful content.

    ``weekly_risk`` is computed as the faithful trailing-7-rows sum of
    ``daily_risk`` per user (the engine's convention), so the ranking by peak
    Weekly_Risk behaves exactly as it would on real data.
    """
    rng = np.random.default_rng(seed)
    days = [_START_DAY + pd.Timedelta(days=d) for d in range(_N_DAYS)]

    users: list[str] = [f"USER{idx:04d}" for idx in range(1, n_normals + 1)]
    insiders: list[str] = [
        f"INSIDER{idx:04d}" for idx in range(1, _N_INSIDERS + 1)
    ]
    all_users = users + insiders
    insider_set = set(insiders)

    reasons_template = [
        {
            "feature": "usb_connect_count",
            "value": 30.0,
            "baseline_mean": 1.5,
            "z_score": 5.0,
        },
        {
            "feature": "after_hours_logons",
            "value": 12.0,
            "baseline_mean": 0.8,
            "z_score": 4.2,
        },
        {
            "feature": "external_recipients",
            "value": 8.0,
            "baseline_mean": 1.1,
            "z_score": 3.6,
        },
    ]

    records: list[dict] = []
    for user in all_users:
        is_insider = user in insider_set
        for day in days:
            if is_insider:
                self_s = float(np.clip(rng.normal(0.7, 0.1), 0.0, 1.0))
                iso_s = float(np.clip(rng.normal(0.6, 0.1), 0.0, 1.0))
                peer_s = float(np.clip(rng.normal(0.5, 0.1), 0.0, 1.0))
                reasons = [dict(entry) for entry in reasons_template]
            else:
                self_s = float(np.clip(rng.normal(0.15, 0.08), 0.0, 1.0))
                iso_s = float(np.clip(rng.normal(0.12, 0.08), 0.0, 1.0))
                peer_s = float(np.clip(rng.normal(0.10, 0.08), 0.0, 1.0))
                reasons = []
            daily = 100.0 * (0.5 * self_s + 0.3 * iso_s + 0.2 * peer_s)
            records.append(
                {
                    "user": user,
                    "day": day,
                    "self_score": self_s,
                    "isolation_score": iso_s,
                    "peer_score": peer_s,
                    "daily_risk": daily,
                    "reasons": reasons,
                }
            )

    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(["user", "day"]).reset_index(drop=True)
    # Weekly_Risk = trailing-7-rows sum of daily_risk per user (engine convention).
    frame["weekly_risk"] = frame.groupby("user", sort=False)["daily_risk"].transform(
        lambda s: s.rolling(window=7, min_periods=1).sum()
    )
    frame["user"] = frame["user"].astype("string")
    frame["day"] = pd.to_datetime(frame["day"])
    # Column order matching the engine's risk.parquet schema.
    return frame[
        [
            "user",
            "day",
            "self_score",
            "isolation_score",
            "peer_score",
            "daily_risk",
            "weekly_risk",
            "reasons",
        ]
    ]


@pytest.fixture(scope="module")
def subset_risk_parquet():
    """Write a Subset-sized ``risk.parquet`` at the configured path, then clean up.

    Backs up any pre-existing ``risk.parquet`` and restores it on teardown; if no
    file existed, removes the one this fixture created. Leaves the repository's
    ``data/processed`` exactly as it was found. Yields ``(config, risk_path,
    n_users)``.
    """
    config = load_config()
    n_normals = int(config.subset.normal_sample_size)
    risk_path = risk_parquet_path(config).resolve()
    risk_path.parent.mkdir(parents=True, exist_ok=True)

    backup_path: Path | None = None
    if risk_path.exists():
        backup_path = risk_path.with_name(risk_path.name + ".bak_test")
        shutil.copy2(risk_path, backup_path)

    try:
        frame = _build_subset_risk_frame(n_normals)
        write_risk(frame, config)
        yield config, risk_path, n_normals + _N_INSIDERS
    finally:
        # Restore the original file, or remove the synthetic one we created.
        if backup_path is not None and backup_path.exists():
            shutil.move(str(backup_path), str(risk_path))
        elif risk_path.exists():
            risk_path.unlink()


def _app_path() -> str:
    """Absolute path to the repository's ``app.py`` (the dashboard entry point)."""
    return str(Path(__file__).resolve().parent.parent / "app.py")


def test_date_and_threshold_controls_render(subset_risk_parquet):
    """Example (Req 8.2): the date range and risk-threshold controls render.

    Runs the real ``app.py`` headlessly and asserts both controls are present —
    at least one ``date_input`` (the date range) and at least one ``slider`` (the
    Daily_Risk threshold) — and that the script rendered without raising.
    """
    _config, _risk_path, _n_users = subset_risk_parquet

    app = AppTest.from_file(_app_path(), default_timeout=60)
    app.run()

    # The script must render cleanly (no uncaught exception bubbling up).
    assert not app.exception, f"app raised: {[str(e) for e in app.exception]}"

    # Date control (Req 8.2): a date range picker is present.
    assert len(app.date_input) >= 1, "expected a date-range control to render"

    # Risk-threshold control (Req 8.2): a Daily_Risk slider is present.
    assert len(app.slider) >= 1, "expected a risk-threshold slider to render"

    # The threshold slider is wired to the Daily_Risk range and starts at 0.
    threshold_slider = app.slider[0]
    assert threshold_slider.value == pytest.approx(0.0)


def test_dashboard_interactive_within_10_seconds(subset_risk_parquet):
    """Integration (Req 8.6): load + first render finish within 10 seconds.

    Measures the two things that gate interactivity on the Subset: reading the
    pre-computed ``risk.parquet`` and running the app's first render. Each must
    complete comfortably within the 10-second budget on a Subset-sized frame.
    """
    config, _risk_path, n_users = subset_risk_parquet

    # (a) Loading the pre-computed risk artifact.
    load_start = time.perf_counter()
    risk_df = load_risk(config)
    load_elapsed = time.perf_counter() - load_start

    # Sanity: the Subset really is Subset-sized and ranks cleanly.
    assert risk_df["user"].nunique() == n_users
    ranking = rank_users(risk_df)
    assert len(ranking) == n_users

    assert load_elapsed < _INTERACTIVE_BUDGET_SECONDS, (
        f"loading risk.parquet took {load_elapsed:.2f}s, "
        f"over the {_INTERACTIVE_BUDGET_SECONDS:.0f}s budget"
    )

    # (b) First full render of the app over the Subset.
    app = AppTest.from_file(_app_path(), default_timeout=60)
    render_start = time.perf_counter()
    app.run()
    render_elapsed = time.perf_counter() - render_start

    assert not app.exception, f"app raised: {[str(e) for e in app.exception]}"
    assert render_elapsed < _INTERACTIVE_BUDGET_SECONDS, (
        f"first render took {render_elapsed:.2f}s, "
        f"over the {_INTERACTIVE_BUDGET_SECONDS:.0f}s budget"
    )
