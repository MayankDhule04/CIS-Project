"""Evaluation example and integration tests (task 5.4).

These are deterministic, seeded example/integration tests (not Hypothesis
property tests). They exercise the Evaluation_Module end to end on a small
synthetic dataset and cover four acceptance clauses:

* **Requirement 7.4** — the report compares exactly the three scoring variants
  (``zscore_only``, ``isolation_only``, ``fused``) side by side.
* **Requirement 7.5 / 7.7** — the written report and its plot files exist, and
  no plain-accuracy metric is ever reported.
* **Requirement 7.1** — ground-truth labels are an evaluation-only concern: they
  never reach the feature, baseline, or model layers. This is checked two ways,
  statically (those layers never name the label machinery) and behaviourally
  (removing ``labels.csv`` changes neither the features nor the risk scores).
* **Requirement 7.6** — a single command (``run_evaluation`` and the
  ``scripts/run_eval.py`` wrapper) produces ``eval_report.md`` and its plots.

Everything is written into ``tempfile.TemporaryDirectory`` locations via
``dataclasses.replace`` on the loaded config, so nothing ever lands in the
repository's own ``reports/`` or ``data/`` directories.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from generate_synthetic import SyntheticGenerator

from ueba.config import Config, load_config
from ueba.evaluation.metrics import load_labels
from ueba.evaluation.report import (
    METRIC_COLUMNS,
    VARIANTS,
    VariantComparison,
    compare_variants,
    run_evaluation,
    write_report,
)
from ueba.features import extract_features
from ueba.scoring import compute_risk, write_risk

# tests/ -> repository root is its parent.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_CONFIG_PATH = _REPO_ROOT / "config.yaml"

# Small-but-sufficient synthetic dataset. n_days comfortably exceeds the
# baseline window (min 30, target 75, max 90) so every user has both a baseline
# period and scored days after the cutoff; the two planted insiders land on days
# after the cutoff so they surface as positive insider user-days.
_N_USERS = 8
_N_DAYS = 100
_SEED = 7
_PLANTS = [
    ("USER0002", 95, "usb_burst"),
    ("USER0005", 90, "after_hours_logons"),
]
_EXPECTED_INSIDERS = {user for user, _day, _kind in _PLANTS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(raw_dir: Path, processed_dir: Path) -> Config:
    """A Config identical to the repo's but pointing at throwaway temp dirs."""
    base = load_config(str(_CONFIG_PATH))
    return dataclasses.replace(
        base,
        raw_data_dir=str(raw_dir),
        processed_dir=str(processed_dir),
    )


def _generate_dataset(raw_dir: Path) -> None:
    """Write a small synthetic dataset with the two planted insiders."""
    gen = SyntheticGenerator(config_path=str(_CONFIG_PATH))
    gen.generate(str(raw_dir), n_users=_N_USERS, n_days=_N_DAYS, seed=_SEED)
    for user, day_offset, kind in _PLANTS:
        gen.plant_anomaly(user, day_offset, kind)


def _build_risk(config: Config) -> pd.DataFrame:
    """Run the feature extractor and Scoring_Engine to produce risk rows."""
    features = extract_features(config)
    risk_df = compute_risk(features, config)
    write_risk(risk_df, config)
    return risk_df


def _package_sources(package: str) -> list[Path]:
    """All ``*.py`` files under ``src/ueba/<package>`` (excluding caches)."""
    pkg_dir = _SRC_DIR / "ueba" / package
    return [p for p in pkg_dir.rglob("*.py") if "__pycache__" not in p.parts]


# ---------------------------------------------------------------------------
# Example 1 — three variants, no accuracy, report + plots exist (7.4/7.5/7.7)
# ---------------------------------------------------------------------------


def test_report_has_three_variants_no_accuracy_and_writes_files():
    """Three variants reported, no accuracy metric, report and plots on disk.

    Validates Requirements 7.4, 7.5, 7.7.

    Builds risk + labels on a small planted dataset, compares the variants, and
    writes the report into a temp reports directory. Asserts the structured
    result has exactly the three variant rows and the four metric columns (no
    accuracy), that the Markdown report names all three variants, and that the
    bar-chart and PR-curve PNGs exist (a PR curve is drawn because the planted
    insiders give positive user-days).

    Accuracy assertion (documented): rather than a blanket text search — which
    would forbid even an explanatory sentence about *omitting* accuracy — we
    assert the robust, structural facts: ``accuracy`` is not a metrics column,
    and no Markdown table row (a line beginning with ``|``) contains the word
    ``accuracy``. That proves no accuracy value is ever reported while still
    allowing prose that explains the omission.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        reports_dir = tmp_path / "reports"

        _generate_dataset(raw_dir)
        config = _make_config(raw_dir, processed_dir)
        risk_df = _build_risk(config)
        labels_df = load_labels(config)

        results = compare_variants(risk_df, labels_df, config)

        # --- structured result: exactly three variants x four metrics -------
        assert isinstance(results, VariantComparison)
        assert list(results.metrics.index) == list(VARIANTS.keys())
        assert set(results.metrics.index) == {
            "zscore_only",
            "isolation_only",
            "fused",
        }
        assert list(results.metrics.columns) == METRIC_COLUMNS
        assert "accuracy" not in {c.lower() for c in results.metrics.columns}

        # The planted insiders must be present as positives for a PR curve.
        assert results.summary["n_insider_users"] == len(_EXPECTED_INSIDERS)
        assert results.summary["n_insider_user_days"] >= len(_PLANTS)

        out_path = reports_dir / "eval_report.md"
        written = write_report(results, out_path=out_path, plots_dir=reports_dir)

        # --- files exist ----------------------------------------------------
        assert written == out_path
        assert out_path.is_file()
        assert (reports_dir / "eval_metrics_bar.png").is_file()
        assert (reports_dir / "eval_pr_curves.png").is_file()

        report_text = out_path.read_text(encoding="utf-8")

        # All three human-readable variant labels appear in the report.
        for label in VARIANTS.values():
            assert label in report_text

        # No accuracy value is reported: no Markdown table cell mentions it.
        table_lines = [
            line for line in report_text.splitlines() if line.lstrip().startswith("|")
        ]
        assert table_lines, "report should contain a Markdown metrics table"
        for line in table_lines:
            assert "accuracy" not in line.lower()

        # The four intended metrics do head the table.
        header = table_lines[0].lower()
        for metric in METRIC_COLUMNS:
            assert metric.lower() in header

        # The report was written inside the temp tree, not the repo's reports/.
        assert tmp_path in out_path.parents


# ---------------------------------------------------------------------------
# Example 2 — label isolation: labels never reach features/model/baselines (7.1)
# ---------------------------------------------------------------------------


def test_labels_never_reach_feature_baseline_or_model_layers():
    """Ground-truth labels are evaluation-only (label isolation).

    Validates Requirement 7.1.

    Two complementary checks:

    * **Static** — the feature, baseline, and model layers must not even name
      the label machinery. Every ``*.py`` under ``src/ueba/features``,
      ``src/ueba/baselines``, and ``src/ueba/models`` is read and asserted to
      contain none of ``labels.csv``, ``load_labels``, ``insider``, or a
      ``ueba.evaluation`` import. This proves the code has no path to labels.
    * **Behavioural** — removing ``labels.csv`` must change nothing those layers
      produce. We extract features and compute risk with the labels file
      present, delete ``labels.csv``, rebuild from scratch, and assert the
      feature table and the risk frame are byte-for-byte identical (training is
      seeded, so the IsolationForest is reproducible). If any layer secretly
      consumed labels, deleting them would perturb the output.
    """
    # --- static check -------------------------------------------------------
    forbidden = ("labels.csv", "load_labels", "insider", "ueba.evaluation")
    for package in ("features", "baselines", "models"):
        for source in _package_sources(package):
            text = source.read_text(encoding="utf-8").lower()
            for token in forbidden:
                assert token.lower() not in text, (
                    f"{source} references '{token}', but the {package} layer "
                    "must never touch ground-truth labels (Req 7.1)."
                )

    # --- behavioural check --------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"

        _generate_dataset(raw_dir)
        labels_path = raw_dir / "labels.csv"
        assert labels_path.is_file(), "generator should have written labels.csv"

        config = _make_config(raw_dir, processed_dir)

        features_with_labels = extract_features(config)
        risk_with_labels = compute_risk(features_with_labels, config)

        # Remove the labels and rebuild everything the model layers see.
        labels_path.unlink()
        assert not labels_path.is_file()

        features_without_labels = extract_features(config)
        risk_without_labels = compute_risk(features_without_labels, config)

        pd.testing.assert_frame_equal(
            features_with_labels.reset_index(drop=True),
            features_without_labels.reset_index(drop=True),
        )
        # ``reasons`` holds lists; assert_frame_equal compares them element-wise.
        pd.testing.assert_frame_equal(
            risk_with_labels.reset_index(drop=True),
            risk_without_labels.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Integration — single command generates eval_report.md and plots (7.6)
# ---------------------------------------------------------------------------


def test_run_evaluation_single_command_writes_report_and_plots():
    """``run_evaluation`` builds risk, compares, and writes the report + plots.

    Validates Requirement 7.6.

    Points a throwaway config at a temp synthetic dataset and calls the one
    entry point with ``rebuild=True`` (so it builds ``risk.parquet`` itself),
    then asserts it returns a ``VariantComparison`` and that the report and both
    plot PNGs were written into the temp output directory.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        out_dir = tmp_path / "out"

        _generate_dataset(raw_dir)
        config = _make_config(raw_dir, processed_dir)

        out_path = out_dir / "eval_report.md"
        results = run_evaluation(
            config=config,
            out_path=out_path,
            plots_dir=out_dir,
            rebuild=True,
        )

        assert isinstance(results, VariantComparison)
        assert list(results.metrics.index) == list(VARIANTS.keys())
        assert out_path.is_file()
        assert (out_dir / "eval_metrics_bar.png").is_file()
        assert (out_dir / "eval_pr_curves.png").is_file()
        # run_evaluation builds and persists risk.parquet into the temp dir.
        assert (processed_dir / "risk.parquet").is_file()


def test_run_eval_script_single_command_offline(tmp_path_factory):
    """The ``scripts/run_eval.py`` one-command path works end to end, offline.

    Validates Requirement 7.6.

    Proves the *shell* one-command invocation (not just the Python function)
    produces the report. A temp ``config.yaml`` is written pointing at a temp
    synthetic dataset and temp processed dir, then ``python scripts/run_eval.py``
    is run as a subprocess with ``PYTHONPATH`` covering the repo root and
    ``src/`` and ``--out`` / ``--plots-dir`` aimed at a temp directory. Asserts
    the process exits 0 and the report and plots exist. No network is used; the
    plotting backend is the offline Agg backend the report module selects.
    """
    tmp_path = tmp_path_factory.mktemp("run_eval_script")
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    out_dir = tmp_path / "out"
    out_path = out_dir / "eval_report.md"
    temp_config = tmp_path / "config.yaml"

    _generate_dataset(raw_dir)

    # A temp config.yaml identical to the repo's but with temp, absolute paths
    # (POSIX-style so YAML stays clean on Windows too).
    base_config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    base_config["paths"]["raw_data_dir"] = raw_dir.as_posix()
    base_config["paths"]["processed_dir"] = processed_dir.as_posix()
    temp_config.write_text(yaml.safe_dump(base_config), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), str(_SRC_DIR), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    completed = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "run_eval.py"),
            "--config",
            str(temp_config),
            "--out",
            str(out_path),
            "--plots-dir",
            str(out_dir),
            "--rebuild",
        ],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert completed.returncode == 0, (
        f"run_eval.py failed (rc={completed.returncode}).\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert out_path.is_file()
    assert (out_dir / "eval_metrics_bar.png").is_file()
    assert (out_dir / "eval_pr_curves.png").is_file()
