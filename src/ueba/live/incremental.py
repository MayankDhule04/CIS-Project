"""Near-real-time (daily micro-batch) incremental scoring.

The core system is batch: it learns each user's baseline from their early
history and scores later days against it. This module turns that into a
*near-real-time* capability at **daily cadence** - the granularity the features
are defined at, and the same cadence commercial UEBA tools score windows on. It
is honest about what it is: it does not score individual events in real time.

The idea:

1. **Freeze** the baselines (self + peer) and the IsolationForest once, from the
   historical data, and persist them (:func:`freeze_baselines`). The design
   already learns baselines from the baseline period only, so freezing them is
   exactly right - later days must be judged against a fixed "normal".
2. As each **new day** of logs arrives, compute that day's user-day features and
   score them against the frozen baselines and model, fuse into daily risk,
   update each user's trailing-7 weekly risk using their prior history, and
   **append** the new rows to ``risk.parquet`` (:func:`incremental_update`).

A polling watcher (``scripts/live_watch.py``) calls :func:`incremental_update`
on an interval, so the dashboard - which reads ``risk.parquet`` - reflects new
days as they land. Everything runs offline with no network calls.

Note on isolation-score scaling: :func:`score_against_baselines` scales the raw
IsolationForest score within the batch it is given (a day's users), consistent
with the batch design's documented, relative scaling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from ..config import Config

# Persisted frozen-baseline bundle (self + peer) alongside the model.
BASELINES_FILENAME = "baselines.joblib"


def baselines_path(config: Config) -> Path:
    """Return ``<processed_dir>/baselines.joblib``."""
    return Path(config.processed_dir) / BASELINES_FILENAME


def save_baselines(self_baseline, peer_baseline, path) -> Path:
    """Persist the self and peer baselines together with joblib."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"self": self_baseline, "peer": peer_baseline}, out)
    return out


def load_baselines(path):
    """Load the ``{"self": ..., "peer": ...}`` baseline bundle written by save."""
    in_path = Path(path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Frozen baselines not found: {in_path}")
    bundle = joblib.load(in_path)
    return bundle["self"], bundle["peer"]


def freeze_baselines(config: Config, user_day_features: Optional[pd.DataFrame] = None):
    """Build and persist the baselines + IsolationForest to score future days.

    Baselines come from each user's baseline period only (handled inside
    :func:`~src.ueba.baselines.build_baselines`), so the frozen "normal" is the
    early-history normal. The model is trained on baseline-period rows too. Both
    are written under ``processed_dir`` so later incremental runs reuse them
    rather than relearning. Returns ``(self_baseline, peer_baseline, model)``.
    """
    from ..baselines import build_baselines
    from ..features import extract_features
    from ..models import default_model_path, save, train_from_features

    feats = user_day_features if user_day_features is not None else extract_features(config)
    self_baseline, peer_baseline, _meta = build_baselines(feats, config)
    model = train_from_features(feats, config)

    save(model, default_model_path(config))
    save_baselines(self_baseline, peer_baseline, baselines_path(config))
    return self_baseline, peer_baseline, model


def _empty_risk() -> pd.DataFrame:
    """An empty risk frame with the canonical columns."""
    from ..scoring.engine import RISK_COLUMNS

    return pd.DataFrame(columns=RISK_COLUMNS)


def score_against_baselines(
    new_features: pd.DataFrame,
    self_baseline,
    peer_baseline,
    model,
    config: Config,
    prior_risk: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Score a batch of new user-day features against FROZEN baselines + model.

    Unlike :func:`~src.ueba.scoring.engine.compute_risk`, this never rebuilds the
    baselines or retrains the model - it reuses the frozen ones, which is what
    makes incremental, day-by-day scoring consistent over time. ``prior_risk``
    (the existing ``risk.parquet``) supplies each user's earlier daily risks so
    the trailing-7 ``weekly_risk`` of the new days is correct. Returns a risk
    frame with the canonical columns.
    """
    from ..models import score as iso_score
    from ..scoring.engine import RISK_COLUMNS, daily_risk, scale_isolation
    from ..scoring.statistical import TOP_K, score_statistical

    if new_features is None or new_features.empty:
        return _empty_risk()

    stat = score_statistical(new_features, self_baseline, peer_baseline)
    stat["isolation_score"] = scale_isolation(iso_score(model, new_features))
    for col in ("self_score", "peer_score", "isolation_score"):
        stat[col] = stat[col].fillna(0.0)
    stat["daily_risk"] = daily_risk(
        stat["self_score"], stat["isolation_score"], stat["peer_score"], config.weights
    )
    stat["reasons"] = stat["self_top_features"].apply(lambda feats: list(feats)[:TOP_K])

    new = stat[
        ["user", "day", "self_score", "isolation_score", "peer_score", "daily_risk", "reasons"]
    ].copy()
    new["day"] = pd.to_datetime(new["day"])

    # Weekly risk = trailing-7 sum per user, using prior history where present.
    if prior_risk is not None and not prior_risk.empty:
        hist = prior_risk[["user", "day", "daily_risk"]].copy()
        hist["day"] = pd.to_datetime(hist["day"])
    else:
        hist = pd.DataFrame(columns=["user", "day", "daily_risk"])
    hist["_is_new"] = False

    tmp_new = new[["user", "day", "daily_risk"]].copy()
    tmp_new["_is_new"] = True

    combined = pd.concat([hist, tmp_new], ignore_index=True)
    combined["user"] = combined["user"].astype("string")
    combined = combined.sort_values(["user", "day"]).reset_index(drop=True)
    combined["weekly_risk"] = combined.groupby("user", sort=False)["daily_risk"].transform(
        lambda s: s.rolling(window=7, min_periods=1).sum()
    )
    weekly = combined.loc[combined["_is_new"], ["user", "day", "weekly_risk"]]

    new["user"] = new["user"].astype("string")
    new = new.merge(weekly, on=["user", "day"], how="left")
    return new[RISK_COLUMNS]


def incremental_update(
    config: Config,
    as_of_day: Optional[object] = None,
    user_day_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Score the new day(s) not yet in ``risk.parquet`` and append them.

    Loads the frozen baselines + model (freezing them first if absent), takes the
    per-user-day features (``user_day_features`` or a fresh extraction), scores
    only the ``(user, day)`` rows not already present in ``risk.parquet``, and
    writes the combined result back. Optionally restricts to a single
    ``as_of_day``. Re-running with no genuinely new days is a no-op (idempotent).
    Returns just the newly scored rows.
    """
    from ..features import extract_features
    from ..models import default_model_path, load as load_model
    from ..scoring.engine import RISK_FILENAME, write_risk

    # Frozen artifacts - freeze on first use.
    feats = user_day_features if user_day_features is not None else extract_features(config)
    model_file = default_model_path(config)
    base_file = baselines_path(config)
    if model_file.is_file() and base_file.is_file():
        model = load_model(model_file)
        self_baseline, peer_baseline = load_baselines(base_file)
    else:
        self_baseline, peer_baseline, model = freeze_baselines(config, feats)

    risk_path = Path(config.processed_dir) / RISK_FILENAME
    prior = pd.read_parquet(risk_path) if risk_path.is_file() else None

    feats = feats.copy()
    feats["day"] = pd.to_datetime(feats["day"])
    if as_of_day is not None:
        target = pd.Timestamp(as_of_day).normalize()
        feats = feats[feats["day"].dt.normalize() == target]

    if prior is not None and not prior.empty:
        seen = set(
            zip(prior["user"].astype("string"), pd.to_datetime(prior["day"]).dt.normalize())
        )
        mask = [
            (u, d) not in seen
            for u, d in zip(feats["user"].astype("string"), feats["day"].dt.normalize())
        ]
        candidates = feats[pd.Series(mask, index=feats.index)]
    else:
        candidates = feats

    if candidates.empty:
        return _empty_risk()

    new_rows = score_against_baselines(
        candidates, self_baseline, peer_baseline, model, config, prior_risk=prior
    )

    combined = new_rows if prior is None or prior.empty else pd.concat([prior, new_rows], ignore_index=True)
    combined = combined.sort_values(["user", "day"]).reset_index(drop=True)
    write_risk(combined, config)
    return new_rows
