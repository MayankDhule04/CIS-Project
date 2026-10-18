"""Tests for the report Limitations text parametrization (data source)."""

from __future__ import annotations

import pandas as pd
import pytest

from ueba.evaluation.report import METRIC_COLUMNS, VariantComparison, write_report


def _results() -> VariantComparison:
    metrics = pd.DataFrame(
        [[0.1, 0.2, 0.3, 0.4]], index=["fused"], columns=METRIC_COLUMNS
    )
    metrics.index.name = "variant"
    return VariantComparison(
        metrics=metrics,
        pr_curves={"fused": None},
        summary={
            "n_users": 5,
            "n_user_days": 10,
            "n_insider_users": 1,
            "n_insider_user_days": 2,
        },
    )


def test_default_is_synthetic(tmp_path):
    path = write_report(_results(), out_path=tmp_path / "r.md", plots_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "**synthetic data**" in text
    assert "real CMU/SEI CERT r4.2 dataset" not in text
    assert "**Data source:** synthetic data" in text


def test_real_source_limitations(tmp_path):
    path = write_report(
        _results(), out_path=tmp_path / "r.md", plots_dir=tmp_path, data_source="real"
    )
    text = path.read_text(encoding="utf-8")
    assert "real CMU/SEI CERT r4.2 dataset" in text
    assert "produced on **synthetic data**" not in text
    assert "dense needles" in text


def test_invalid_source_raises(tmp_path):
    with pytest.raises(ValueError):
        write_report(
            _results(), out_path=tmp_path / "r.md", plots_dir=tmp_path, data_source="bogus"
        )
