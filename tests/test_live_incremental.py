"""Tests for near-real-time daily incremental scoring (CI-safe, synthetic)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from ueba.config import load_config
from ueba.features import FEATURE_COLUMNS
from ueba.live import (
    baselines_path,
    freeze_baselines,
    incremental_update,
    load_baselines,
    score_against_baselines,
)
from ueba.models import default_model_path


def _feature_frame(n_users: int = 4, n_days: int = 45, seed: int = 0) -> pd.DataFrame:
    """Build an in-memory user-day feature frame spanning more than min_days."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2010-01-04")
    roles = ["Analyst", "Engineer"]
    rows = []
    for u in range(n_users):
        user = f"USER{u:04d}"
        role = roles[u % len(roles)]
        for d in range(n_days):
            row = {"user": user, "day": start + pd.Timedelta(days=d), "role": role}
            for col in FEATURE_COLUMNS:
                row[col] = int(rng.integers(0, 5))
            rows.append(row)
    return pd.DataFrame(rows, columns=["user", "day", "role", *FEATURE_COLUMNS])


def _config(tmp_path):
    base = load_config("config.yaml")
    proc = tmp_path / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    return dataclasses.replace(base, processed_dir=str(proc))


def test_freeze_persists_model_and_baselines(tmp_path):
    cfg = _config(tmp_path)
    feats = _feature_frame()
    freeze_baselines(cfg, feats)
    assert default_model_path(cfg).is_file()
    assert baselines_path(cfg).is_file()
    self_b, peer_b = load_baselines(baselines_path(cfg))
    assert not self_b.mean.empty
    assert not peer_b.mean.empty


def test_incremental_update_appends_then_idempotent(tmp_path):
    cfg = _config(tmp_path)
    feats = _feature_frame()

    first = incremental_update(cfg, user_day_features=feats)
    assert len(first) == len(feats)  # nothing existed, so all days scored

    risk = pd.read_parquet(default_model_path(cfg).parent / "risk.parquet")
    assert len(risk) == len(feats)
    assert {"user", "day", "daily_risk", "weekly_risk", "reasons"}.issubset(risk.columns)

    # Re-running with the same data scores nothing new (idempotent).
    second = incremental_update(cfg, user_day_features=feats)
    assert second.empty
    risk2 = pd.read_parquet(default_model_path(cfg).parent / "risk.parquet")
    assert len(risk2) == len(feats)


def test_incremental_scores_only_the_new_day(tmp_path):
    cfg = _config(tmp_path)
    feats = _feature_frame(n_days=45)
    last_day = feats["day"].max()
    history = feats[feats["day"] < last_day]
    new_day = feats[feats["day"] == last_day]

    incremental_update(cfg, user_day_features=history)
    new_rows = incremental_update(cfg, user_day_features=feats)

    # Only the final day's user-days are newly scored.
    assert len(new_rows) == len(new_day)
    assert (pd.to_datetime(new_rows["day"]).dt.normalize() == last_day.normalize()).all()


def test_weekly_risk_uses_prior_history(tmp_path):
    cfg = _config(tmp_path)
    cfg = dataclasses.replace(cfg)
    # One user, 8 consecutive days; freeze on it, then check weekly = trailing-7.
    feats = _feature_frame(n_users=1, n_days=40, seed=3)
    self_b, peer_b, model = freeze_baselines(cfg, feats)

    scored = score_against_baselines(feats, self_b, peer_b, model, cfg, prior_risk=None)
    scored = scored.sort_values("day").reset_index(drop=True)
    # The last row's weekly_risk equals the sum of the last 7 daily_risk values.
    expected = scored["daily_risk"].tail(7).sum()
    assert abs(float(scored["weekly_risk"].iloc[-1]) - float(expected)) < 1e-6
