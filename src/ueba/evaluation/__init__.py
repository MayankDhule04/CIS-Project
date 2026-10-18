"""Evaluation layer: detection metrics, variant comparison, and reporting.

This package is the system's single, deliberate entry point for ground-truth
labels (design rule "labels are evaluation-only", Requirement 7.1). It exposes
the Evaluation_Module's detection-quality metrics and the small helpers that
load labels and shape risk data for them:

* :func:`recall_at_k` / :func:`precision_at_k` — user-level detection quality at
  a cutoff ``k`` (``k = 20`` and ``k = 50``; Requirement 7.2).
* :func:`pr_auc` — precision-recall AUC at the user-day level (Requirement 7.3).
* :func:`load_labels` / :func:`insider_users` / :func:`user_day_label_vector` /
  :func:`rank_users_by_weekly_risk` — load the ground-truth labels and turn
  ``risk.parquet`` into the ranking and the aligned score/label vectors the
  metrics consume.

It also exposes the variant comparison and reporting half of the module:

* :func:`compare_variants` — grade the z-score-only, IsolationForest-only, and
  fused variants side by side (Requirement 7.4).
* :func:`write_report` — write ``reports/eval_report.md`` plus static plots
  (Requirement 7.5).
* :func:`run_evaluation` — the single-command entry point that builds/loads
  ``risk.parquet``, compares the variants, and writes the report
  (Requirement 7.6). Also runnable as ``python -m ueba.evaluation`` or
  ``python scripts/run_eval.py``.

Plain accuracy is intentionally absent (Requirement 7.7): with insiders a tiny
minority it would be misleadingly high and is never reported.
"""

from .metrics import (
    LABELS_FILENAME,
    insider_users,
    load_labels,
    pr_auc,
    precision_at_k,
    rank_users_by_weekly_risk,
    recall_at_k,
    user_day_label_vector,
)
from .report import (
    METRIC_COLUMNS,
    VARIANTS,
    VariantComparison,
    compare_variants,
    run_evaluation,
    variant_user_day_score,
    write_report,
)

__all__ = [
    "LABELS_FILENAME",
    "recall_at_k",
    "precision_at_k",
    "pr_auc",
    "load_labels",
    "insider_users",
    "user_day_label_vector",
    "rank_users_by_weekly_risk",
    "VARIANTS",
    "METRIC_COLUMNS",
    "VariantComparison",
    "variant_user_day_score",
    "compare_variants",
    "write_report",
    "run_evaluation",
]
