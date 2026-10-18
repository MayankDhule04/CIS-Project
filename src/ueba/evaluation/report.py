"""Variant comparison and evaluation report generation (Evaluation_Module).

This module is the reporting half of the Evaluation_Module. The metrics
themselves live in :mod:`src.ueba.evaluation.metrics` (the single, deliberate
entry point for ground-truth labels); this module *uses* those metrics to grade
three scoring recipes side by side and to write a human-readable report with
charts. It never touches features, baselines, or models, and — like the rest of
the evaluation layer — it never computes plain accuracy (Requirement 7.7).

The three variants (Requirement 7.4)
------------------------------------
The fused risk score blends three signals: a *self* z-score, an *IsolationForest*
score, and a *peer* z-score. To show what each ingredient contributes, we grade
three recipes against the same ground truth:

* ``zscore_only`` — the **statistical** part of the fusion only. We renormalise
  the self and peer weights so they sum to one and combine the two statistical
  scores with them::

      zscore_score = (self_w * self_score + peer_w * peer_score) / (self_w + peer_w)

  Both ``self_score`` and ``peer_score`` already lie in ``0``–``1``, so the
  blended z-score does too. This is the documented definition of the "z-score
  only" user-day score; the IsolationForest contribution is dropped entirely.
* ``isolation_only`` — the IsolationForest's ``isolation_score`` alone (already
  scaled to ``0``–``1`` by the Scoring_Engine).
* ``fused`` — the real fused risk: ``daily_risk`` at the user-day level and its
  trailing-7 roll-up at the user level (i.e. ``weekly_risk``).

Fair, identical aggregation across variants
--------------------------------------------
User-level recall/precision need **one score per user** from a per-user-day
score. To keep the three-way comparison fair, every variant is aggregated the
*same* way: for each user we take a trailing-7-row rolling sum of that variant's
user-day score (ordered by day) and then rank users by their per-user **maximum**
of that rolling sum. This is exactly the roll-up the Scoring_Engine already uses
for ``weekly_risk``, so for the ``fused`` variant this aggregation coincides with
the canonical ``weekly_risk``-based ranking produced by
:func:`~src.ueba.evaluation.metrics.rank_users_by_weekly_risk`; the other two
variants are ranked by the identical rule applied to their own user-day score.

User-day-level PR-AUC (Requirement 7.3) uses each variant's user-day score
directly, aligned row-for-row to the ground-truth user-day labels via
:func:`~src.ueba.evaluation.metrics.user_day_label_vector`.

Plots (Requirement 7.5) — offline, no network
----------------------------------------------
Charts are rendered with **Matplotlib** using the non-interactive ``Agg``
backend and saved as static PNG files, which embed cleanly in Markdown. This
choice is deliberate: Matplotlib is already installed and needs no browser, no
image-export engine (Plotly's PNG export needs ``kaleido``, which is not
installed), and makes no network calls — satisfying the offline guarantee
(Requirement 10.3). Matplotlib is pinned in ``requirements.txt`` for this
reason. Two files are written: a grouped bar chart of every metric across the
three variants, and an overlay of the three precision-recall curves.

Single-command entry point (Requirement 7.6)
---------------------------------------------
:func:`run_evaluation` loads ``risk.parquet`` (building it from the synthetic
pipeline if it is absent), loads the labels, compares the variants, and writes
``reports/eval_report.md`` plus the plots — all from one command::

    python scripts/run_eval.py

(The package also exposes ``python -m ueba.evaluation`` when ``src/`` is on the
import path.) See :func:`run_evaluation` for the exact invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .metrics import (
    insider_users,
    load_labels,
    pr_auc,
    precision_at_k,
    rank_users_by_weekly_risk,
    recall_at_k,
    user_day_label_vector,
)

__all__ = [
    "VARIANTS",
    "METRIC_COLUMNS",
    "VariantComparison",
    "variant_user_day_score",
    "compare_variants",
    "write_report",
    "run_evaluation",
]

# The three scoring recipes compared side by side (Requirement 7.4). The key is
# the stable identifier used in the metrics table and plots; the value is the
# human-readable label used in the report.
VARIANTS: dict[str, str] = {
    "zscore_only": "Z-score only",
    "isolation_only": "IsolationForest only",
    "fused": "Fused (self + iso + peer)",
}

# The metric columns reported for every variant. Plain accuracy is intentionally
# absent (Requirement 7.7).
METRIC_COLUMNS: list[str] = ["recall@20", "recall@50", "precision@20", "pr_auc"]

# The user-level ranking cutoffs (Requirement 7.2).
RECALL_KS = (20, 50)
PRECISION_K = 20

# Length (in rows, ordered by day) of the trailing window used to aggregate a
# variant's user-day score up to one score per user — the same convention the
# Scoring_Engine uses for weekly_risk.
WEEKLY_WINDOW = 7

# Default locations, relative to the current working directory, matching the
# design's "reports/eval_report.md".
DEFAULT_REPORT_PATH = Path("reports") / "eval_report.md"
BAR_CHART_FILENAME = "eval_metrics_bar.png"
PR_CURVE_FILENAME = "eval_pr_curves.png"


# Limitations text depends on which dataset the run used (synthetic vs
# real CERT r4.2); the paragraph must describe whichever was actually scored.
VALID_DATA_SOURCES = ("synthetic", "real")

_LIMITATIONS_SYNTHETIC = 'These results are produced on **synthetic data** generated to match the CERT release 4.2 schema, not the real 20 GB dataset. The planted anomalies are deliberately large and clean relative to the simulated normal behaviour, so the detector separates them more easily than it would separate real insiders from real noise. The numbers here are therefore **optimistic** and should be read as a demonstration that the pipeline and the three scoring variants behave as designed, not as an estimate of real-world detection quality. Metrics also degrade gracefully (reported as `n/a`) when a run has no positive labels.'

_LIMITATIONS_REAL = 'These results are produced on the **real CMU/SEI CERT r4.2 dataset**. r4.2 is the "dense needles" release: it interleaves an unusually high volume of injected malicious activity compared with a real deployment, so insiders are easier to separate here than they would be in production, where malicious behaviour is far rarer and subtler. The detector is **unsupervised and ranking-based**: it surfaces anomalous user-days for an analyst to review rather than issuing automated verdicts, and the ground-truth labels are used only to score the ranking, never for training or feature construction. Results depend on a stable baseline period; users with insufficient early history are excluded from scoring. As the variant comparison shows, the IsolationForest is a weaker standalone signal than the self z-score on this data. Metrics also degrade gracefully (reported as `n/a`) when a run has no positive labels.'

_LIMITATIONS_BY_SOURCE = {
    "synthetic": _LIMITATIONS_SYNTHETIC,
    "real": _LIMITATIONS_REAL,
}


@dataclass(frozen=True)
class VariantComparison:
    """The side-by-side result of comparing the three scoring variants.

    Attributes:
        metrics: A DataFrame indexed by variant id (``zscore_only``,
            ``isolation_only``, ``fused``) with one column per metric in
            :data:`METRIC_COLUMNS`. This is the primary structured result
            (rows = variants, columns = metrics); it never contains accuracy.
        pr_curves: Per-variant precision-recall curve points for plotting,
            keyed by variant id. Each value is a dict with ``precision`` and
            ``recall`` arrays and the scalar ``average_precision`` (equal to the
            variant's ``pr_auc``). A variant with no positive labels maps to
            ``None`` (its curve is undefined).
        summary: Small counts describing the evaluated data (number of users,
            user-days, insider users, insider user-days) for the report header.
    """

    metrics: pd.DataFrame
    pr_curves: dict[str, Optional[dict]] = field(default_factory=dict)
    summary: dict[str, int] = field(default_factory=dict)


def variant_user_day_score(
    risk_df: pd.DataFrame, variant: str, config: Config
) -> np.ndarray:
    """Return a variant's per-user-day score, aligned to ``risk_df`` rows.

    Args:
        risk_df: The fused risk frame (``risk.parquet``), carrying ``self_score``,
            ``peer_score``, ``isolation_score``, and ``daily_risk`` columns.
        variant: One of the keys of :data:`VARIANTS`.
        config: The loaded config, used for the self/peer fusion weights when
            building the ``zscore_only`` blend.

    Returns:
        A float64 array of user-day scores in the same row order as ``risk_df``.

    The ``zscore_only`` score is the statistical part of the fusion with the
    self and peer weights renormalised to sum to one (see the module docstring);
    ``isolation_only`` is the raw ``isolation_score``; ``fused`` is ``daily_risk``.
    """
    if variant == "zscore_only":
        self_w = float(config.weights.self_w)
        peer_w = float(config.weights.peer_w)
        denom = self_w + peer_w
        self_s = risk_df["self_score"].to_numpy(dtype="float64")
        peer_s = risk_df["peer_score"].to_numpy(dtype="float64")
        if denom <= 0.0:
            # Degenerate weights: fall back to a plain mean of the two.
            return (self_s + peer_s) / 2.0
        return (self_w * self_s + peer_w * peer_s) / denom
    if variant == "isolation_only":
        return risk_df["isolation_score"].to_numpy(dtype="float64")
    if variant == "fused":
        return risk_df["daily_risk"].to_numpy(dtype="float64")
    raise ValueError(
        f"Unknown variant '{variant}'. Valid variants: {list(VARIANTS)}."
    )


def _rank_users_for_variant(
    risk_df: pd.DataFrame, user_day_score: np.ndarray
) -> list[str]:
    """Rank users most-risky first using the uniform trailing-7 aggregation.

    Builds a per-(user, day) frame from ``user_day_score``, takes each user's
    trailing-7-row rolling sum ordered by day, names it ``weekly_risk``, and
    hands it to :func:`~src.ueba.evaluation.metrics.rank_users_by_weekly_risk`
    so every variant is ranked by the identical rule (per-user max of the
    rolling-7 sum, ties broken by user id). For the ``fused`` variant this
    reproduces the canonical ``weekly_risk`` ranking.
    """
    frame = pd.DataFrame(
        {
            "user": risk_df["user"].astype("string").to_numpy(),
            "day": pd.to_datetime(risk_df["day"]).to_numpy(),
            "_score": np.asarray(user_day_score, dtype="float64"),
        }
    )
    frame = frame.sort_values(["user", "day"]).reset_index(drop=True)
    frame["weekly_risk"] = frame.groupby("user", sort=False)["_score"].transform(
        lambda series: series.rolling(window=WEEKLY_WINDOW, min_periods=1).sum()
    )
    return rank_users_by_weekly_risk(frame)


def compare_variants(
    risk_df: pd.DataFrame, labels_df: pd.DataFrame, config: Config
) -> VariantComparison:
    """Grade the z-only, iso-only, and fused variants side by side.

    For each variant this computes, against the same ground truth:

    * user-level ``recall@20``, ``recall@50``, and ``precision@20`` from a
      ranking produced by the uniform trailing-7 aggregation (so the three
      rankings are directly comparable); and
    * user-day-level ``pr_auc`` from the variant's user-day score aligned
      row-for-row to the user-day labels.

    Args:
        risk_df: The fused risk frame (``risk.parquet``).
        labels_df: The ground-truth labels as returned by
            :func:`~src.ueba.evaluation.metrics.load_labels`.
        config: The loaded config (supplies the fusion weights).

    Returns:
        A :class:`VariantComparison` whose ``metrics`` DataFrame has rows =
        variants and columns = :data:`METRIC_COLUMNS`. Plain accuracy is never
        included (Requirement 7.7). Metrics degrade gracefully (NaN / 0.0) when
        there are no positive labels, exactly as the underlying metrics do.
    """
    # Import here so the module has no hard dependency on sklearn unless a PR
    # curve is actually drawn (and so an empty-label run never needs it).
    from sklearn.metrics import precision_recall_curve

    insiders = insider_users(labels_df)

    # User-day labels aligned to risk_df's row order (the second element is the
    # 0/1 vector; the first is daily_risk, which we don't use here).
    _daily, y_true = user_day_label_vector(risk_df, labels_df)

    rows: dict[str, list[float]] = {}
    pr_curves: dict[str, Optional[dict]] = {}

    for variant in VARIANTS:
        user_day_score = variant_user_day_score(risk_df, variant, config)
        ranking = _rank_users_for_variant(risk_df, user_day_score)

        r20 = recall_at_k(ranking, insiders, RECALL_KS[0])
        r50 = recall_at_k(ranking, insiders, RECALL_KS[1])
        p20 = precision_at_k(ranking, insiders, PRECISION_K)
        auc = pr_auc(user_day_score, y_true)
        rows[variant] = [r20, r50, p20, auc]

        # Precision-recall curve points for the overlay plot (only defined when
        # at least one positive user-day exists).
        if int(np.sum(y_true == 1)) > 0:
            precision, recall, _thresholds = precision_recall_curve(
                y_true, user_day_score
            )
            pr_curves[variant] = {
                "precision": precision,
                "recall": recall,
                "average_precision": auc,
            }
        else:
            pr_curves[variant] = None

    metrics = pd.DataFrame.from_dict(
        rows, orient="index", columns=METRIC_COLUMNS
    )
    metrics.index.name = "variant"

    summary = {
        "n_users": int(risk_df["user"].astype("string").nunique()),
        "n_user_days": int(len(risk_df)),
        "n_insider_users": int(len(insiders)),
        "n_insider_user_days": int(np.sum(y_true == 1)),
    }

    return VariantComparison(metrics=metrics, pr_curves=pr_curves, summary=summary)


def _format_metric(value: float) -> str:
    """Render a metric for the Markdown table, showing ``n/a`` for NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.3f}"


def _write_bar_chart(metrics: pd.DataFrame, path: Path) -> None:
    """Write a grouped bar chart of every metric across the three variants."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive, offline-safe backend
    import matplotlib.pyplot as plt

    variants = list(metrics.index)
    labels = [VARIANTS.get(v, v) for v in variants]
    n_metrics = len(METRIC_COLUMNS)
    n_variants = len(variants)

    x = np.arange(n_metrics)
    width = 0.8 / max(n_variants, 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, variant in enumerate(variants):
        heights = np.nan_to_num(
            metrics.loc[variant, METRIC_COLUMNS].to_numpy(dtype="float64"),
            nan=0.0,
        )
        offset = (i - (n_variants - 1) / 2) * width
        ax.bar(x + offset, heights, width, label=labels[i])

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_COLUMNS)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection metrics by scoring variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_pr_curves(pr_curves: dict[str, Optional[dict]], path: Path) -> bool:
    """Write an overlay of the three precision-recall curves.

    Returns ``True`` if at least one curve was drawn (so the report only links
    the plot when it exists), ``False`` when every variant lacked positives.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drawn = False
    fig, ax = plt.subplots(figsize=(7, 6))
    for variant, curve in pr_curves.items():
        if curve is None:
            continue
        label = (
            f"{VARIANTS.get(variant, variant)} "
            f"(AP={curve['average_precision']:.3f})"
        )
        ax.plot(curve["recall"], curve["precision"], label=label)
        drawn = True

    if not drawn:
        plt.close(fig)
        return False

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_title("User-day precision-recall curves by variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def write_report(
    results: VariantComparison,
    out_path: "str | Path" = DEFAULT_REPORT_PATH,
    plots_dir: "str | Path | None" = None,
    data_source: str = "synthetic",
) -> Path:
    """Write the Markdown evaluation report and its plots (Requirement 7.5).

    Args:
        results: The :class:`VariantComparison` from :func:`compare_variants`.
        out_path: Where to write the Markdown report. Defaults to
            ``reports/eval_report.md``.
        plots_dir: Directory for the PNG plots. Defaults to the report's own
            directory, so the Markdown links the plots by bare filename.

    Returns:
        The path to the written Markdown report.

    The report contains the side-by-side metrics table (rows = variants,
    columns = the four metrics — never accuracy), embeds the bar chart and, when
    there are positive labels, the precision-recall curves, and closes with a
    short limitations paragraph noting that synthetic data makes these numbers
    optimistic.
    """
    if data_source not in VALID_DATA_SOURCES:
        raise ValueError(
            f"Unknown data_source {data_source!r}; expected one of {VALID_DATA_SOURCES}."
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(plots_dir) if plots_dir is not None else out_path.parent
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = results.metrics

    # --- plots -------------------------------------------------------------
    bar_path = plots_dir / BAR_CHART_FILENAME
    _write_bar_chart(metrics, bar_path)
    bar_rel = Path(__import__("os").path.relpath(bar_path, out_path.parent))

    pr_path = plots_dir / PR_CURVE_FILENAME
    has_pr = _write_pr_curves(results.pr_curves, pr_path)
    pr_rel = (
        Path(__import__("os").path.relpath(pr_path, out_path.parent))
        if has_pr
        else None
    )

    # --- markdown ----------------------------------------------------------
    summary = results.summary
    lines: list[str] = []
    lines.append("# UEBA Detection Quality — Evaluation Report")
    lines.append("")
    lines.append(
        "This report grades the insider-threat detector against ground-truth "
        "labels and compares three scoring variants side by side. A plain "
        "correct-rate metric is deliberately omitted (Requirement 7.7): insider "
        "user-days are a tiny minority, so such a metric would look high while "
        "revealing nothing about whether insiders are actually caught. The "
        "metrics below are all ranking- and detection-focused instead."
    )
    lines.append("")
    source_label = (
        "real CMU/SEI CERT r4.2 dataset"
        if data_source == "real"
        else "synthetic data (CERT r4.2 schema)"
    )
    lines.append(f"**Data source:** {source_label}.")
    lines.append("")
    if summary:
        lines.append(
            f"**Evaluated data:** {summary.get('n_users', 0)} users, "
            f"{summary.get('n_user_days', 0)} user-days, "
            f"{summary.get('n_insider_users', 0)} insider users, "
            f"{summary.get('n_insider_user_days', 0)} insider user-days."
        )
        lines.append("")

    lines.append("## Variant comparison")
    lines.append("")
    lines.append(
        "Each variant is ranked for the user-level metrics by the same "
        "trailing-7 aggregation (per-user maximum of a 7-day rolling sum of that "
        "variant's user-day score), so the three columns are directly "
        "comparable. PR-AUC is measured at the user-day level."
    )
    lines.append("")

    # Header row: variant + the four metrics.
    header = "| Variant | " + " | ".join(METRIC_COLUMNS) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(METRIC_COLUMNS)) + " |"
    lines.append(header)
    lines.append(sep)
    for variant in metrics.index:
        label = VARIANTS.get(variant, variant)
        cells = [_format_metric(metrics.loc[variant, col]) for col in METRIC_COLUMNS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append(f"![Detection metrics by variant]({bar_rel.as_posix()})")
    lines.append("")
    if pr_rel is not None:
        lines.append(f"![User-day precision-recall curves]({pr_rel.as_posix()})")
        lines.append("")
    else:
        lines.append(
            "_Precision-recall curves were not drawn because the evaluated data "
            "contained no positive insider user-days._"
        )
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(_LIMITATIONS_BY_SOURCE[data_source])
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def run_evaluation(
    config: "Config | None" = None,
    out_path: "str | Path" = DEFAULT_REPORT_PATH,
    plots_dir: "str | Path | None" = None,
    rebuild: bool = False,
    data_source: str = "synthetic",
) -> VariantComparison:
    """Single-command evaluation: build/load risk, compare, and write the report.

    This is the Evaluation_Module's one-command entry point (Requirement 7.6).
    Run it from the repository root with::

        python scripts/run_eval.py

    (equivalently ``python -m ueba.evaluation`` when ``src/`` is on the import
    path). It:

    1. loads ``config.yaml`` if ``config`` is not supplied;
    2. reads ``<processed_dir>/risk.parquet`` — or, if the file is absent (or
       ``rebuild`` is ``True``), builds it from scratch on the current data by
       running the feature extractor and the Scoring_Engine, then persists it;
    3. loads the ground-truth labels (the only place labels enter the system);
    4. compares the three variants with :func:`compare_variants`; and
    5. writes ``reports/eval_report.md`` and its plots with :func:`write_report`.

    Args:
        config: An optional pre-loaded :class:`~src.ueba.config.Config`.
        out_path: Where to write the Markdown report.
        plots_dir: Directory for the plots (defaults to the report's directory).
        rebuild: Force rebuilding ``risk.parquet`` even if it already exists.

    Returns:
        The :class:`VariantComparison` that was written to the report.
    """
    if config is None:
        config = load_config()

    risk_df = _load_or_build_risk(config, rebuild=rebuild)
    labels_df = load_labels(config)

    results = compare_variants(risk_df, labels_df, config)
    report_path = write_report(
        results, out_path=out_path, plots_dir=plots_dir, data_source=data_source
    )
    print(f"[run_evaluation] Wrote evaluation report to {report_path}")
    return results


def _load_or_build_risk(config: Config, rebuild: bool = False) -> pd.DataFrame:
    """Read ``risk.parquet`` if present, otherwise build and persist it.

    Building runs the feature extractor over the raw/synthetic CSVs and then the
    Scoring_Engine's :func:`~src.ueba.scoring.compute_risk`, writing the result
    to ``<processed_dir>/risk.parquet`` so later runs (and the dashboard) can
    reuse it. Imports of the feature and scoring layers are local so that merely
    reading an existing ``risk.parquet`` never pulls in the model stack.
    """
    from ..scoring import RISK_FILENAME

    risk_path = Path(config.processed_dir) / RISK_FILENAME
    if risk_path.is_file() and not rebuild:
        return pd.read_parquet(risk_path)

    # Build from scratch on the current (synthetic) data.
    from ..features import extract_features
    from ..scoring import compute_risk, write_risk

    features = extract_features(config)
    risk_df = compute_risk(features, config)
    write_risk(risk_df, config)
    return risk_df
