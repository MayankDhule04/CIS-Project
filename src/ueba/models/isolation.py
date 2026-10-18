"""Isolation_Model — a machine-learned detector of days that don't fit.

The statistical scorer (task 3.5) asks "how far is this day from a person's own
or their peers' averages?" This module adds a second, complementary opinion: an
unsupervised scikit-learn ``IsolationForest`` that learns the *shape* of calm,
early-period behaviour across all the features at once and then flags days that
sit in sparse, easy-to-isolate corners of that space.

Two rules from the design drive everything here:

* **Train on the baseline period only.** The forest is fit using exactly the
  user-days dated on or before each user's Baseline_Cutoff, and every row dated
  after the cutoff is excluded from training (design Property 13, Req 6.1/6.2).
  :func:`select_training_rows` performs that time split by reusing the very same
  per-user cutoff logic the baselines use, so the forest and the statistical
  baselines always agree on what "baseline period" means.
* **Scale features to 0–1 before fitting.** A ``MinMaxScaler`` is fit on the
  training rows and bundled *with* the forest, so scoring applies the identical
  transform. This keeps features on a common footing and makes a saved model
  fully self-contained.

What this module deliberately does **not** do: it does not scale the raw
anomaly score into the 0–1 ``Isolation_Score``, fuse scores, compute daily or
weekly risk, or build reason records. Those belong to the Scoring_Engine (task
4.3). Here, :meth:`IsolationModel.score` / :func:`score` return a **raw anomaly
score per row where higher means more anomalous** (the negated
``IsolationForest.score_samples`` output); the Scoring_Engine takes it from
there.

Persistence
-----------
A trained model is a small bundle — the fitted forest, its fitted scaler, and
the exact feature-column order — persisted together with :func:`save` and
restored with :func:`load`. Because the bundle is self-contained, a reloaded
model scores identically to the in-memory one (design Property 18, Req 6.10).
The default on-disk location is ``<processed_dir>/isolation_model.joblib``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from ..config import Config
from ..features import FEATURE_COLUMNS

__all__ = [
    "IsolationModel",
    "DEFAULT_MODEL_FILENAME",
    "default_model_path",
    "select_training_rows",
    "train",
    "train_from_features",
    "score",
    "save",
    "load",
]

# The forest is saved here under the processed directory by default (Req 6.9).
DEFAULT_MODEL_FILENAME = "isolation_model.joblib"

# Defaults for the IsolationForest. A fixed random_state makes training (and
# therefore saved models and their scores) reproducible run to run.
DEFAULT_N_ESTIMATORS = 200
DEFAULT_RANDOM_STATE = 42


@dataclass
class IsolationModel:
    """A self-contained IsolationForest bundle: forest + scaler + columns.

    Bundling the fitted ``estimator`` with the ``scaler`` that was fit on the
    same training rows, plus the exact ``feature_columns`` order used to build
    the training matrix, guarantees that scoring always rebuilds the input the
    same way the model was trained — whether the model is used in memory or
    reloaded from disk.

    Attributes
    ----------
    estimator:
        The fitted scikit-learn :class:`~sklearn.ensemble.IsolationForest`.
    scaler:
        The :class:`~sklearn.preprocessing.MinMaxScaler` fit on the training
        rows; applied to every scoring input so features land in 0–1.
    feature_columns:
        The ordered feature names the model was trained on. Scoring selects and
        orders input columns to match exactly this list.
    """

    estimator: IsolationForest
    scaler: MinMaxScaler
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def _feature_matrix(self, user_days: pd.DataFrame) -> np.ndarray:
        """Build the scaled 0–1 feature matrix for ``user_days``.

        Selects the model's feature columns in their trained order (treating any
        missing column as all-zeros), coerces them to numeric, fills gaps with
        ``0``, and applies the bundled scaler. The result is exactly the matrix
        the forest expects.
        """
        frame = pd.DataFrame(index=getattr(user_days, "index", None))
        for col in self.feature_columns:
            if col in user_days.columns:
                frame[col] = pd.to_numeric(user_days[col], errors="coerce")
            else:
                frame[col] = 0.0
        matrix = frame.fillna(0.0).to_numpy(dtype="float64")
        return self.scaler.transform(matrix)

    def score(self, user_days: pd.DataFrame) -> np.ndarray:
        """Return a raw anomaly score per input row (higher = more anomalous).

        The score is the negated ``IsolationForest.score_samples`` value, so the
        ordering is intuitive: larger numbers mark days that are easier for the
        forest to isolate and therefore more anomalous. Exactly one finite value
        is returned per row of ``user_days``, in input order. This is the *raw*
        score; scaling it into the 0–1 Isolation_Score is the Scoring_Engine's
        job (task 4.3).
        """
        if user_days is None or len(user_days) == 0:
            return np.empty(0, dtype="float64")
        matrix = self._feature_matrix(user_days)
        return -self.estimator.score_samples(matrix)


def default_model_path(config: Config) -> Path:
    """Return the default model path ``<processed_dir>/isolation_model.joblib``."""
    return Path(config.processed_dir) / DEFAULT_MODEL_FILENAME


def _training_matrix(
    baseline_features: pd.DataFrame, feature_columns: list[str]
) -> np.ndarray:
    """Build the unscaled training matrix from the given feature columns.

    Missing columns are treated as all-zeros and non-numeric/NaN values are
    coerced to ``0`` so training never fails on a sparse or partial frame.
    """
    frame = pd.DataFrame(index=baseline_features.index)
    for col in feature_columns:
        if col in baseline_features.columns:
            frame[col] = pd.to_numeric(baseline_features[col], errors="coerce")
        else:
            frame[col] = 0.0
    return frame.fillna(0.0).to_numpy(dtype="float64")


def select_training_rows(
    user_day_features: pd.DataFrame, config: Config
) -> pd.DataFrame:
    """Return exactly the baseline-period rows the forest may train on.

    Delegates the time split to the baselines layer's
    :func:`~src.ueba.baselines.builder.select_baseline_rows`, which computes
    each user's inclusive Baseline_Cutoff and keeps only the rows on or before
    it. Every row dated after a user's cutoff is therefore excluded (design
    Property 13, Req 6.1/6.2). The returned frame carries the same columns as
    the input, restricted to in-period rows.
    """
    # Imported here to avoid a circular import at module load time
    # (baselines imports from features, not from models).
    from ..baselines import select_baseline_rows

    in_period_all, _meta = select_baseline_rows(user_day_features, config)
    return in_period_all


def train(
    baseline_features: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    contamination: "str | float" = "auto",
) -> IsolationModel:
    """Fit an :class:`IsolationModel` on already-selected baseline rows.

    ``baseline_features`` must already be restricted to the Baseline_Period
    (see :func:`select_training_rows`); this function fits the 0–1 scaler and
    the IsolationForest on exactly those rows and returns a self-contained
    bundle. Features are scaled to 0–1 before fitting. Use
    :func:`train_from_features` if you have a full, unfiltered feature set and
    want the baseline split applied for you.

    Raises
    ------
    ValueError
        If ``baseline_features`` is empty (there is nothing to learn from).
    """
    if baseline_features is None or len(baseline_features) == 0:
        raise ValueError(
            "Cannot train the Isolation_Model on an empty baseline feature set."
        )

    columns = list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
    raw_matrix = _training_matrix(baseline_features, columns)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw_matrix)

    estimator = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    estimator.fit(scaled)

    return IsolationModel(estimator=estimator, scaler=scaler, feature_columns=columns)


def train_from_features(
    user_day_features: pd.DataFrame,
    config: Config,
    feature_columns: Optional[list[str]] = None,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    contamination: "str | float" = "auto",
) -> IsolationModel:
    """Select baseline-period rows and train on them end-to-end.

    This is the convenience entry point: it first restricts
    ``user_day_features`` to each user's Baseline_Period via
    :func:`select_training_rows` (so no post-cutoff row can leak into training),
    then fits the model via :func:`train`. The returned bundle is ready to
    :func:`save` and :func:`score`.
    """
    in_period = select_training_rows(user_day_features, config)
    return train(
        in_period,
        feature_columns=feature_columns,
        n_estimators=n_estimators,
        random_state=random_state,
        contamination=contamination,
    )


def score(model: IsolationModel, user_days: pd.DataFrame) -> np.ndarray:
    """Return raw anomaly scores for ``user_days`` (higher = more anomalous).

    Thin module-level wrapper over :meth:`IsolationModel.score` matching the
    design's ``score(model, user_days) -> np.ndarray`` signature.
    """
    return model.score(user_days)


def save(model: IsolationModel, path: "str | Path") -> Path:
    """Persist the model bundle (forest + scaler + columns) with joblib.

    Writes the whole :class:`IsolationModel` to ``path`` (creating parent
    directories as needed) so a later :func:`load` restores a model that scores
    identically (design Property 18, Req 6.9/6.10). The conventional location is
    ``<processed_dir>/isolation_model.joblib`` — see :func:`default_model_path`.
    Returns the path written.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    return out_path


def load(path: "str | Path") -> IsolationModel:
    """Load a model bundle previously written by :func:`save`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    in_path = Path(path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Isolation_Model file not found: {in_path}")
    return joblib.load(in_path)
