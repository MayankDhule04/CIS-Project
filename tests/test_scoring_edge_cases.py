"""Edge-case (example) tests for the Statistical_Scorer.

Task 3.7 — edge cases for scoring, Requirement 5.6.

These are deterministic example/edge tests (not Hypothesis properties). They
pin down three corners of the statistical scorer that the property tests only
touch in general:

1. A planted USB burst on a scoring-period day produces a z-score greater than
   4 on the spiked ``usb_connect_count`` feature (Requirement 5.6). This drives
   the real generator -> feature extractor -> baseline -> scorer path end to
   end, planting the anomaly *after* the baseline cutoff so the spike is scored
   against a clean baseline.
2. A zero-variance baseline feature yields ``z = 0`` even when the user-day
   value differs from the baseline mean — no division by zero, and the overall
   score stays finite.
3. A short-history user (fewer than ``config.baseline.min_days`` active days) is
   flagged ``sufficient == False`` so scoring can exclude them, while a
   long-history user is flagged sufficient.
"""

from __future__ import annotations

import dataclasses
import math
import tempfile
from datetime import date, timedelta

import pandas as pd

from src.ueba.baselines import build_baselines, build_self_baseline
from src.ueba.config import load_config
from src.ueba.features import FEATURE_COLUMNS, extract_features
from src.ueba.scoring.statistical import self_score

from generate_synthetic import SyntheticGenerator

# Repository root -> the config.yaml both the generator and loader read. The
# conftest puts the repo root, ``src``, and ``scripts`` on ``sys.path``; we use
# an absolute path so the test never depends on the process working directory.
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = str(_REPO_ROOT / "config.yaml")

# The generator's CERT release 4.2 calendar start (see _DEFAULT_START_DATE).
_START_DATE = date(2010, 1, 4)


def _config_for(raw_dir: str, processed_dir: str):
    """Load the real config but point the data dirs at the test's temp dirs."""
    config = load_config(_CONFIG_PATH)
    return dataclasses.replace(
        config, raw_data_dir=raw_dir, processed_dir=processed_dir
    )


# ---------------------------------------------------------------------------
# Edge 1: planted USB spike -> z-score > 4 on the spiked feature (Req 5.6)
# ---------------------------------------------------------------------------


def test_planted_usb_spike_scores_zscore_above_four():
    """A planted USB burst yields z > 4 on usb_connect_count (Requirement 5.6).

    Generate a long run so the target user builds a baseline of low/zero USB
    usage, plant a ``usb_burst`` on a day well past the baseline cutoff (a
    scoring-period day), re-extract features from the rewritten CSVs, build
    baselines from the in-period rows, and self-score the planted user-day. The
    spiked ``usb_connect_count`` feature must surface in the top features with a
    z-score above 4.
    """
    n_days = 120
    # Baseline cutoff is the first ~75 days (target_days clamped to 30..90), so
    # day 100 is firmly inside the scoring period.
    day_offset = 100
    user_id = "USER0002"
    planted_day = pd.Timestamp(_START_DATE + timedelta(days=day_offset))

    with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as processed_dir:
        config = _config_for(raw_dir, processed_dir)

        gen = SyntheticGenerator(config_path=_CONFIG_PATH)
        gen.generate(raw_dir, n_users=3, n_days=n_days, seed=42)
        # plant_anomaly rewrites the CSVs, so features must be extracted after.
        gen.plant_anomaly(user_id, day_offset, "usb_burst")

        user_day = extract_features(config)

        # Self baseline is built from in-period rows only; the planted day lives
        # in the scoring period and is therefore excluded from the baseline.
        self_baseline, _peer, meta = build_baselines(user_day, config)

    # The planted user has plenty of early history, so their baseline is trusted.
    assert bool(meta.loc[user_id, "sufficient"]) is True

    planted_rows = user_day[
        (user_day["user"] == user_id) & (user_day["day"] == planted_day)
    ]
    assert len(planted_rows) == 1, "expected exactly one planted user-day row"
    planted_row = planted_rows.iloc[0]
    # Sanity: the burst really planted a large USB connect count.
    assert planted_row["usb_connect_count"] >= 30

    score, top_features = self_score(planted_row, self_baseline)

    # The score must be a finite number in [0, 1].
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0

    by_feature = {tf["feature"]: tf for tf in top_features}
    assert "usb_connect_count" in by_feature, (
        f"usb_connect_count missing from top features: {list(by_feature)}"
    )
    assert by_feature["usb_connect_count"]["z_score"] > 4.0, (
        f"planted USB spike z-score {by_feature['usb_connect_count']['z_score']} "
        "was not greater than 4"
    )


# ---------------------------------------------------------------------------
# Edge 2: zero-variance baseline feature -> z = 0, finite score
# ---------------------------------------------------------------------------


def test_zero_variance_feature_yields_zero_zscore():
    """A feature that never varied in the baseline scores z = 0 (no divide-by-0).

    Build a self baseline where ``usb_connect_count`` is constant (zero
    variance) while ``logon_count`` varies. Score a user-day whose
    ``usb_connect_count`` differs sharply from the baseline mean: the scorer
    must define that feature's z-score as exactly 0 rather than dividing by a
    zero standard deviation, and the overall score must stay finite.
    """
    # 40 in-period days for one user: usb_connect_count constant at 0 (std == 0),
    # logon_count varying so at least one feature carries a real deviation.
    n_rows = 40
    in_period = pd.DataFrame(
        {
            "user": ["U1"] * n_rows,
            "usb_connect_count": [0] * n_rows,
            "logon_count": [1 + (i % 3) for i in range(n_rows)],
        }
    )

    self_baseline = build_self_baseline(in_period, min_days=30)
    # The baseline really is zero-variance on the USB feature.
    assert float(self_baseline.std.loc["U1", "usb_connect_count"]) == 0.0

    # A user-day that differs on the zero-variance feature.
    user_day = {"user": "U1", "usb_connect_count": 10, "logon_count": 9}
    score, top_features = self_score(user_day, self_baseline)

    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0

    by_feature = {tf["feature"]: tf for tf in top_features}
    assert "usb_connect_count" in by_feature
    assert by_feature["usb_connect_count"]["z_score"] == 0.0, (
        "zero-variance feature should score z = 0, got "
        f"{by_feature['usb_connect_count']['z_score']}"
    )


# ---------------------------------------------------------------------------
# Edge 3: short-history user flagged insufficient (excluded from scoring)
# ---------------------------------------------------------------------------


def test_short_history_user_flagged_insufficient():
    """A user with fewer than min_days active days is flagged insufficient.

    Construct a feature table with one short-history user (10 active days) and
    one long-history user (40 active days), both inside the baseline window.
    ``build_baselines`` must flag the short user ``sufficient == False`` (so
    scoring can exclude them) and the long user ``sufficient == True``.
    """
    raw_dir = "unused"
    processed_dir = "unused"
    config = _config_for(raw_dir, processed_dir)
    min_days = config.baseline.min_days  # 30 by default

    short_days = [_START_DATE + timedelta(days=i) for i in range(10)]
    long_days = [_START_DATE + timedelta(days=i) for i in range(40)]

    def _rows(user: str, role: str, days: list[date]) -> list[dict]:
        rows = []
        for d in days:
            row = {"user": user, "day": pd.Timestamp(d), "role": role}
            for feature in FEATURE_COLUMNS:
                row[feature] = 0
            rows.append(row)
        return rows

    frame = pd.DataFrame(
        _rows("SHORT", "Analyst", short_days) + _rows("LONG", "Analyst", long_days)
    )

    _self, _peer, meta = build_baselines(frame, config)

    assert meta.loc["SHORT", "n_days"] == 10
    assert meta.loc["LONG", "n_days"] == 40
    assert bool(meta.loc["SHORT", "sufficient"]) is False
    assert bool(meta.loc["LONG", "sufficient"]) is True

    # The convention: insufficient users form the excluded-from-scoring set.
    excluded = set(meta.index[~meta["sufficient"]])
    assert "SHORT" in excluded
    assert "LONG" not in excluded

    # The self baseline carries the same per-user sufficiency flag.
    assert _self.sufficient is not None
    assert bool(_self.sufficient.loc["SHORT"]) is False
    assert bool(_self.sufficient.loc["LONG"]) is True
