"""Evaluation_Module metrics — grade the detector against ground truth.

This module is the *only* place in the whole UEBA system that reads the
ground-truth insider labels. Nothing in feature extraction, baseline building,
or model training imports anything from here, and this module never writes a
label back into those layers. That one-way flow is the design's third rule —
"labels are evaluation-only" (Requirement 7.1) — and keeping label loading
physically inside the evaluation package is how we enforce it.

What it computes
----------------
Three detection-quality metrics, each matching a reference definition so the
numbers mean exactly what a researcher expects (design Property 19):

* :func:`recall_at_k` — of all the real insider *users*, what fraction landed in
  the top ``k`` of the risk ranking. Reported at ``k = 20`` and ``k = 50``
  (Requirement 7.2).
* :func:`precision_at_k` — of the top ``k`` ranked *users*, what fraction were
  real insiders. Reported at ``k = 20`` (Requirement 7.2).
* :func:`pr_auc` — the area under the precision-recall curve at the
  *user-day* level, i.e. how well the per-day risk score separates insider
  user-days from the rest (Requirement 7.3).

We deliberately never compute plain accuracy (Requirement 7.7). Insider days are
a tiny minority, so accuracy would look great by calling everything "normal" and
would hide the detector's real behaviour. There is no ``accuracy`` function in
this module by design.

Reference definitions
---------------------
For a ranking ``r`` (user ids, most-risky first), an insider-user set ``L``, and
a cutoff ``k``::

    recall@k    = |{top-k users} ∩ L| / |L|
    precision@k = |{top-k users} ∩ L| / k

``pr_auc`` is the standard **average precision** estimator of PR-AUC, computed
with scikit-learn's :func:`sklearn.metrics.average_precision_score`. Average
precision summarises the precision-recall curve as the precision at each
threshold weighted by the increase in recall it brings; it is the usual,
threshold-free way to score PR-AUC and avoids the optimistic bias of trapezoidal
interpolation.

Design decisions (documented so they are not surprises)
-------------------------------------------------------
* **Ranking order is authoritative for ties.** :func:`recall_at_k` and
  :func:`precision_at_k` take ``ranking`` as an already-ordered sequence and
  honour that order exactly; any ties among equal scores must already be broken
  by whoever built the sequence. The convenience builder
  :func:`rank_users_by_weekly_risk` breaks ties deterministically: higher
  Weekly_Risk first, then user id ascending, so the same input always yields the
  same ranking.
* **One score per user for the user-level ranking.** A user is represented by
  their *maximum* Weekly_Risk across all their days — their single riskiest week
  — because user-level recall/precision ask "did we surface this person at all",
  and their peak week is the strongest evidence we have for them.
* **Graceful degradation on empty labels.** With no positive labels the metrics
  are undefined rather than zero. :func:`recall_at_k` and :func:`pr_auc` return
  ``NaN`` (a 0/0 ratio has no meaningful value) and :func:`precision_at_k`
  returns ``0.0`` (no true positive is possible, and its denominator ``k`` is
  still valid); all three emit a :func:`warnings.warn` first so a report can
  still be generated without raising.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ..config import Config

# The ground-truth labels file the Synthetic_Generator writes into the raw data
# directory. Columns: user, day (YYYY-MM-DD), insider (1 for insider user-days).
LABELS_FILENAME = "labels.csv"

# The per-user-day risk column used as the score for user-day-level PR-AUC.
_USER_DAY_SCORE_COLUMN = "daily_risk"

# The per-user risk column collapsed to one value per user for user-level ranks.
_USER_LEVEL_SCORE_COLUMN = "weekly_risk"


def _as_label_set(labels: Iterable[str]) -> set[str]:
    """Return a plain ``set`` of insider user ids from any iterable."""
    return set(labels)


def recall_at_k(ranking: Sequence[str], labels: Iterable[str], k: int) -> float:
    """Fraction of insider users that appear in the top ``k`` of the ranking.

    Args:
        ranking: User ids ordered most-risky first. The order is taken as given;
            ties are assumed to be already broken by the caller.
        labels: The insider user ids (the user-level positives). May be any
            iterable; duplicates are ignored.
        k: How many of the top-ranked users to look at (e.g. 20 or 50).

    Returns:
        ``|top-k ∩ insiders| / |insiders|`` as a float in ``0``–``1``. Returns
        ``NaN`` (after a warning) when there are no insider labels, since the
        ratio is then undefined, or when ``k`` is not positive.

    A user id that appears more than once in ``ranking`` is counted at most once,
    so a malformed ranking cannot inflate recall above 1.0.
    """
    insiders = _as_label_set(labels)
    if not insiders:
        warnings.warn(
            "recall_at_k called with an empty insider label set; "
            "recall is undefined, returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    if k <= 0:
        warnings.warn(
            f"recall_at_k called with non-positive k={k}; returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")

    top_k = set(ranking[:k])
    hits = len(top_k & insiders)
    return hits / len(insiders)


def precision_at_k(ranking: Sequence[str], labels: Iterable[str], k: int) -> float:
    """Fraction of the top ``k`` ranked users that are real insiders.

    Args:
        ranking: User ids ordered most-risky first (order taken as given).
        labels: The insider user ids (the user-level positives).
        k: How many of the top-ranked users to score (e.g. 20).

    Returns:
        ``|top-k ∩ insiders| / k`` as a float in ``0``–``1``. The denominator is
        always ``k`` as in the reference definition, so if ``ranking`` is shorter
        than ``k`` the missing slots count as non-insiders (precision is
        penalised rather than silently rebased on a smaller denominator). Returns
        ``NaN`` (after a warning) when ``k`` is not positive, and ``0.0`` (after
        a warning) when there are no insider labels, since no true positive is
        then possible.
    """
    if k <= 0:
        warnings.warn(
            f"precision_at_k called with non-positive k={k}; returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")

    insiders = _as_label_set(labels)
    if not insiders:
        warnings.warn(
            "precision_at_k called with an empty insider label set; "
            "no true positive is possible, returning 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0

    top_k = set(ranking[:k])
    hits = len(top_k & insiders)
    return hits / k


def pr_auc(scores: Iterable[float], labels: Iterable[int]) -> float:
    """Precision-recall area under the curve at the user-day level.

    Args:
        scores: The per-user-day risk score (higher means more suspicious),
            array-like and aligned one-to-one with ``labels``.
        labels: The per-user-day ground truth, ``1`` for an insider user-day and
            ``0`` otherwise, aligned one-to-one with ``scores``.

    Returns:
        The average-precision estimate of PR-AUC as a float in ``0``–``1``.
        Returns ``NaN`` (after a warning) when there are no positive labels,
        since the curve is undefined without a single insider user-day.

    Raises:
        ValueError: If ``scores`` and ``labels`` have different lengths.

    This uses :func:`sklearn.metrics.average_precision_score`, the standard,
    threshold-free PR-AUC estimator. It is computed here at the user-day level,
    so every scored day is one sample.
    """
    score_arr = np.asarray(list(scores), dtype=float)
    label_arr = np.asarray(list(labels), dtype=float)

    if score_arr.shape[0] != label_arr.shape[0]:
        raise ValueError(
            "pr_auc requires scores and labels of equal length "
            f"(got {score_arr.shape[0]} scores and {label_arr.shape[0]} labels)."
        )

    n_positive = int(np.sum(label_arr == 1))
    if n_positive == 0:
        warnings.warn(
            "pr_auc called with no positive user-day labels; "
            "PR-AUC is undefined, returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")

    return float(average_precision_score(label_arr, score_arr))


# ---------------------------------------------------------------------------
# Label loading and alignment helpers — the single entry point for ground truth.
# ---------------------------------------------------------------------------


def load_labels(config: Config) -> pd.DataFrame:
    """Load the ground-truth insider labels from the raw data directory.

    Reads ``labels.csv`` (written by the Synthetic_Generator, or shipped with
    the real dataset) from ``config.raw_data_dir`` and returns a DataFrame with
    columns ``user`` (string), ``day`` (datetime64), and ``insider`` (int).

    If the file is absent the function degrades gracefully: it warns and returns
    an empty DataFrame with those columns, so an evaluation run can still proceed
    (every metric will then take its empty-label path). This is the only loader
    of label data in the system; no other layer imports it.
    """
    labels_path = Path(config.raw_data_dir) / LABELS_FILENAME
    columns = ["user", "day", "insider"]

    if not labels_path.is_file():
        warnings.warn(
            f"Labels file not found at {labels_path}; "
            "returning an empty label set.",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame({"user": [], "day": [], "insider": []}).astype(
            {"user": "string", "insider": "int64"}
        )

    frame = pd.read_csv(labels_path)
    for required in columns:
        if required not in frame.columns:
            raise ValueError(
                f"Labels file {labels_path} is missing the '{required}' column."
            )
    frame = frame[columns].copy()
    frame["user"] = frame["user"].astype("string")
    frame["day"] = pd.to_datetime(frame["day"])
    frame["insider"] = frame["insider"].astype("int64")
    return frame


def insider_users(labels_df: pd.DataFrame) -> set[str]:
    """Derive the user-level insider set: any user flagged on any day.

    A user counts as an insider for user-level recall/precision if they have at
    least one insider user-day (``insider == 1``). Returns an empty set for an
    empty label frame.
    """
    if labels_df.empty:
        return set()
    flagged = labels_df.loc[labels_df["insider"] == 1, "user"]
    return {str(user) for user in flagged.tolist()}


def user_day_label_vector(
    risk_df: pd.DataFrame, labels_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Align risk rows to 0/1 user-day labels for :func:`pr_auc`.

    Builds two parallel arrays, one per row of ``risk_df`` in its existing order:

    * ``scores`` — the row's ``daily_risk`` (the user-day score PR-AUC ranks).
    * ``y_true`` — ``1`` if that ``(user, day)`` is a positive insider user-day
      in ``labels_df`` (``insider == 1``), else ``0``.

    Days are matched on calendar day regardless of how each source stores the
    timestamp, so a datetime risk column and a string label column line up.
    Returns ``(scores, y_true)`` as NumPy arrays suitable for passing straight
    into :func:`pr_auc`.
    """
    scores = risk_df[_USER_DAY_SCORE_COLUMN].to_numpy(dtype=float)

    risk_users = risk_df["user"].astype("string")
    risk_days = pd.to_datetime(risk_df["day"]).dt.strftime("%Y-%m-%d")
    risk_keys = list(zip(risk_users.tolist(), risk_days.tolist()))

    if labels_df.empty:
        positive_keys: set[tuple[str, str]] = set()
    else:
        positives = labels_df.loc[labels_df["insider"] == 1].copy()
        pos_users = positives["user"].astype("string")
        pos_days = pd.to_datetime(positives["day"]).dt.strftime("%Y-%m-%d")
        positive_keys = set(zip(pos_users.tolist(), pos_days.tolist()))

    y_true = np.array(
        [1 if key in positive_keys else 0 for key in risk_keys], dtype=int
    )
    return scores, y_true


def rank_users_by_weekly_risk(risk_df: pd.DataFrame) -> list[str]:
    """Rank users most-risky first by their peak Weekly_Risk.

    Collapses ``risk_df`` to one value per user — the user's maximum
    ``weekly_risk`` across all their days — then orders users by that value
    descending. Ties are broken deterministically by user id ascending, so the
    same risk data always yields the same ranking. The returned list is the
    ``ranking`` argument expected by :func:`recall_at_k` and
    :func:`precision_at_k`.
    """
    if risk_df.empty:
        return []
    per_user = (
        risk_df.groupby(risk_df["user"].astype("string"))[_USER_LEVEL_SCORE_COLUMN]
        .max()
        .reset_index()
    )
    # Sort by weekly_risk descending, then user ascending for a stable tie-break.
    per_user = per_user.sort_values(
        by=[_USER_LEVEL_SCORE_COLUMN, "user"], ascending=[False, True]
    )
    return [str(user) for user in per_user["user"].tolist()]
