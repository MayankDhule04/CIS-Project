"""Statistical_Scorer — how surprising was this day, against self and peers?

The baselines layer already learned what "ordinary" looks like for each person
(their *self* baseline) and for each job-role group (their *peer* baseline), as
a mean and standard deviation per behavioural feature. This module turns a
single user-day into two anomaly scores in the range ``0``–``1``:

* ``self_score`` — how far the day stands out from *that same user's* own
  normal, comparing the day against the user's self baseline.
* ``peer_score`` — how far the day stands out from the user's *role group's*
  normal, comparing the day against the role's peer baseline.

Scoring rule (design.md "Scoring Algorithm" + Requirements 5.3/5.4, Property 12)
-------------------------------------------------------------------------------
For each feature we compute a z-score against the chosen baseline::

    z = (value - baseline_mean) / baseline_std

and then apply three deliberate conventions:

1. **Zero-variance rule.** When ``baseline_std == 0`` (a feature that never
   varied over the baseline period) the z-score is *defined* as ``0``. There is
   no deviation signal and we must never divide by zero. This mirrors the
   baseline builder, which stores such features' std as ``0.0``.

2. **Cap at 5.** Each z-score is clamped into ``[-5, +5]``. The design caps the
   magnitude ``|z|`` at 5 so a single wild day cannot dominate; we cap
   *symmetrically* so a stored z-score for a reason record is always bounded,
   whether the day was far above or far below normal.

3. **Top-3, scaled by /5.** Anomaly is about standing *out* — doing more than
   usual — so we keep the three features with the largest (most positive)
   z-scores, average those three capped z-scores, and divide by 5 to map the
   ``[0, 5]`` anomaly range onto ``[0, 1]``. If a day is entirely at or below
   normal the averaged top-3 could be ``<= 0``; we clamp the final score into
   ``[0, 1]`` so "nothing unusual" reads as a score near ``0``. When a baseline
   has fewer than three comparable features we simply average the ones we have.

The same three features we scored are also returned as the day's *top features*
— each with its raw ``value``, the ``baseline_mean`` it was compared against,
and its capped ``z_score``. These become the ``self_top_features`` /
``peer_top_features`` list-of-struct columns in ``scores_stat.parquet`` and feed
the later Reason_Record (task 4, Requirement 6.7).

Only features present in *both* the user-day and the baseline are considered,
so a baseline missing a column (or a user-day lacking one) never raises.

What this module does **not** do: it does not train the IsolationForest, scale
its scores, or fuse self/isolation/peer into a daily risk. That is phase 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa

from ..config import Config
from ..features import FEATURE_COLUMNS

__all__ = [
    "Z_CAP",
    "SCORES_STAT_FILENAME",
    "self_score",
    "peer_score",
    "score_statistical",
    "write_scores_stat",
]

# The magnitude each z-score is capped at, and the divisor that scales the
# capped anomaly range [0, Z_CAP] onto the [0, 1] score range.
Z_CAP = 5.0

# The artifact this scorer writes.
SCORES_STAT_FILENAME = "scores_stat.parquet"

# How many top features contribute to a score and to the reason record.
TOP_K = 3

# Explicit Arrow types so the list-of-struct columns round-trip exactly even
# when some rows carry an empty top-features list (which would otherwise give
# pyarrow nothing to infer the struct shape from).
_FEATURE_STRUCT = pa.struct(
    [
        ("feature", pa.string()),
        ("value", pa.float64()),
        ("baseline_mean", pa.float64()),
        ("z_score", pa.float64()),
    ]
)

_SCORES_STAT_SCHEMA = pa.schema(
    [
        ("user", pa.string()),
        ("day", pa.timestamp("ns")),
        ("self_score", pa.float64()),
        ("peer_score", pa.float64()),
        ("self_top_features", pa.list_(_FEATURE_STRUCT)),
        ("peer_top_features", pa.list_(_FEATURE_STRUCT)),
    ]
)


def _as_mapping(user_day: Any) -> Mapping[str, Any]:
    """Return a plain name->value mapping for one user-day.

    Accepts a ``dict``, a pandas ``Series`` (a row from ``df.iterrows`` or
    ``df.loc[i]``), or a one-row ``DataFrame``. Anything else is a programming
    error and raises ``TypeError`` so the caller learns immediately.
    """
    if isinstance(user_day, Mapping):
        return user_day
    if isinstance(user_day, pd.Series):
        return user_day.to_dict()
    if isinstance(user_day, pd.DataFrame):
        if len(user_day) != 1:
            raise TypeError(
                "A DataFrame user-day must have exactly one row; "
                f"got {len(user_day)}."
            )
        return user_day.iloc[0].to_dict()
    raise TypeError(
        "user_day must be a dict, a pandas Series, or a one-row DataFrame; "
        f"got {type(user_day).__name__}."
    )


def _capped_z(value: float, mean: float, std: float) -> float:
    """Return the z-score of ``value`` vs ``(mean, std)``, capped to [-5, 5].

    Applies the zero-variance rule (``std == 0`` -> z of ``0``, never a divide
    by zero) and the symmetric cap at :data:`Z_CAP`.
    """
    if std == 0.0 or not pd.notna(std):
        return 0.0
    z = (value - mean) / std
    if not pd.notna(z):
        return 0.0
    return max(-Z_CAP, min(Z_CAP, z))


def _score_against(
    user_day: Mapping[str, Any], group_key: Any, baseline: Any
) -> "tuple[float, list[dict]]":
    """Score one user-day against one baseline group (a user or a role).

    Looks up ``group_key`` in the baseline's ``mean``/``std`` tables, computes a
    capped z-score for every feature shared by the user-day and the baseline,
    keeps the three largest, and returns ``(score, top_features)`` where the
    score is the mean of those capped z-scores divided by :data:`Z_CAP` and
    clamped into ``[0, 1]``. When the group is unknown (e.g. a short-history
    user excluded from baselines, or a role with no peer baseline) the score is
    ``NaN`` and ``top_features`` is empty, so the caller can skip the row.
    """
    mean_df: pd.DataFrame = baseline.mean
    std_df: pd.DataFrame = baseline.std

    if group_key is None or group_key not in mean_df.index:
        return float("nan"), []

    mean_row = mean_df.loc[group_key]
    std_row = std_df.loc[group_key]

    features = [
        f
        for f in FEATURE_COLUMNS
        if f in mean_df.columns and f in user_day and pd.notna(user_day[f])
    ]
    if not features:
        return float("nan"), []

    scored = []
    for feature in features:
        value = float(user_day[feature])
        baseline_mean = float(mean_row[feature])
        z = _capped_z(value, baseline_mean, float(std_row[feature]))
        scored.append(
            {
                "feature": feature,
                "value": value,
                "baseline_mean": baseline_mean,
                "z_score": z,
            }
        )

    # Largest (most positive) deviations first; stable on feature name so the
    # choice among equal z-scores is deterministic.
    scored.sort(key=lambda item: (-item["z_score"], item["feature"]))
    top_features = scored[:TOP_K]

    mean_top_z = sum(item["z_score"] for item in top_features) / len(top_features)
    score = max(0.0, min(1.0, mean_top_z / Z_CAP))
    return score, top_features


def self_score(user_day: Any, self_baseline: Any) -> "tuple[float, list[dict]]":
    """Score a user-day against *its own* user's self baseline.

    ``user_day`` is a single row (dict, Series, or one-row DataFrame) carrying a
    ``user`` field and the feature columns. The user id selects the baseline
    group. Returns ``(self_score, top_features)`` as described in
    :func:`_score_against`: the score is in ``[0, 1]`` and ``top_features`` is
    the top-3 contributing features, each a dict with ``feature``, ``value``,
    ``baseline_mean``, and ``z_score``.
    """
    row = _as_mapping(user_day)
    return _score_against(row, row.get("user"), self_baseline)


def peer_score(user_day: Any, peer_baseline: Any) -> "tuple[float, list[dict]]":
    """Score a user-day against its *role group's* peer baseline.

    Like :func:`self_score`, but the ``role`` field selects the baseline group,
    so the day is compared against how the user's peers normally behave. Returns
    ``(peer_score, top_features)`` with the score in ``[0, 1]``.
    """
    row = _as_mapping(user_day)
    return _score_against(row, row.get("role"), peer_baseline)


def score_statistical(
    user_day_features: pd.DataFrame,
    self_baseline: Any,
    peer_baseline: Any,
) -> pd.DataFrame:
    """Score every user-day against its self and peer baselines.

    Walks ``user_day_features`` (one row per ``(user, day)``, with ``role`` and
    the feature columns) and produces a frame with columns ``user``, ``day``,
    ``self_score``, ``peer_score``, ``self_top_features``, and
    ``peer_top_features``. The two ``*_top_features`` columns hold a list of
    dicts (list-of-struct) per row — the top-3 contributing features with their
    value, baseline mean, and capped z-score. Rows whose user or role is absent
    from the respective baseline carry a ``NaN`` score and an empty list.
    """
    records: list[dict] = []
    for _, row in user_day_features.iterrows():
        s_score, s_top = self_score(row, self_baseline)
        p_score, p_top = peer_score(row, peer_baseline)
        records.append(
            {
                "user": row["user"],
                "day": row["day"],
                "self_score": s_score,
                "peer_score": p_score,
                "self_top_features": s_top,
                "peer_top_features": p_top,
            }
        )

    columns = [
        "user",
        "day",
        "self_score",
        "peer_score",
        "self_top_features",
        "peer_top_features",
    ]
    frame = pd.DataFrame(records, columns=columns)
    if not frame.empty:
        frame["user"] = frame["user"].astype("string")
        frame["day"] = pd.to_datetime(frame["day"])
    return frame


def write_scores_stat(df: pd.DataFrame, config: Config) -> Path:
    """Write statistical scores to ``<processed_dir>/scores_stat.parquet``.

    Uses an explicit Arrow schema so the ``self_top_features`` /
    ``peer_top_features`` list-of-struct columns persist and read back exactly,
    even for rows whose top-features list is empty. Returns the output path.
    """
    processed_dir = Path(config.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / SCORES_STAT_FILENAME

    table = pa.Table.from_pandas(
        df, schema=_SCORES_STAT_SCHEMA, preserve_index=False
    )
    import pyarrow.parquet as pq

    pq.write_table(table, out_path)
    return out_path
