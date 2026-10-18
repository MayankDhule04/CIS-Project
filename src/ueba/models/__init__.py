"""Models layer: IsolationForest train, save, load, and score.

Exposes the Isolation_Model: an unsupervised scikit-learn ``IsolationForest``
trained only on baseline-period user-days (dated on or before each user's
Baseline_Cutoff), with its features scaled to 0–1. The fitted forest, its
scaler, and the feature-column order travel together as a self-contained
:class:`IsolationModel` bundle so a model saved to
``<processed_dir>/isolation_model.joblib`` reloads and scores identically.

The module returns *raw* anomaly scores (higher = more anomalous); turning
those into the 0–1 Isolation_Score and fusing scores into risk is the
Scoring_Engine's job (task 4.3). See :mod:`src.ueba.models.isolation`.
"""

from .isolation import (
    DEFAULT_MODEL_FILENAME,
    IsolationModel,
    default_model_path,
    load,
    save,
    score,
    select_training_rows,
    train,
    train_from_features,
)

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
