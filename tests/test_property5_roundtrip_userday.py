"""Property-based test for ``user_day.parquet`` artifact round-tripping.

Feature: ueba-insider-threat-detection, Property 5: Artifact persistence
round-trips.

This test exercises the ingestion pipeline's persistence guarantee: whatever
aggregated per-user-day frame ``ingest`` writes to ``user_day.parquet`` must
read back from disk unchanged. The frame ``ingest`` returns is, by contract,
the same object it writes, so reading the parquet file back and comparing it to
the returned frame is a faithful round-trip check.

Each Hypothesis example builds a fresh set of CERT 4.2-shaped raw CSVs with the
``SyntheticGenerator`` (varied user count, day span, and seed), points a copy of
the project config at isolated temp directories via :func:`dataclasses.replace`,
runs ``ingest``, and asserts the on-disk frame equals the returned frame with
matching column order and dtypes.

Every example creates its own ``tempfile.TemporaryDirectory`` inside the test
body — the function-scoped ``tmp_path`` fixture is deliberately avoided because
pairing it with ``@given`` trips a Hypothesis health check. This test lives
entirely in this file and adds no shared fixtures.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from generate_synthetic import SyntheticGenerator
from src.ueba.config import load_config
from src.ueba.ingestion.pipeline import USER_DAY_FILENAME, ingest

# Repository root -> the project's config.yaml. Passed explicitly so neither the
# generator nor the config loader depends on the process working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")

# Loaded once; each example derives an isolated copy with dataclasses.replace.
_BASE_CONFIG = load_config(_CONFIG_PATH)


@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    n_users=st.integers(min_value=1, max_value=4),
    n_days=st.integers(min_value=1, max_value=12),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_user_day_parquet_round_trips(n_users, n_days, seed):
    """Property 5: user_day.parquet reads back equal to the written frame.

    Feature: ueba-insider-threat-detection, Property 5: Artifact persistence
    round-trips. Validates Requirements 2.3.

    For an arbitrary aggregated frame produced by ``ingest`` and written to
    ``user_day.parquet``, reading the file back with pandas yields a frame equal
    to the one ``ingest`` returned (same rows, column order, and dtypes).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_dir = Path(tmp_dir) / "raw"
        processed_dir = Path(tmp_dir) / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Lay down CERT 4.2-shaped raw CSVs for this example's parameters.
        gen = SyntheticGenerator(config_path=_CONFIG_PATH)
        gen.generate(str(raw_dir), n_users=n_users, n_days=n_days, seed=seed)

        # Redirect only the paths; keep the real column_map so ingestion reads
        # the generator's columns exactly as it would in production.
        cfg = dataclasses.replace(
            _BASE_CONFIG,
            raw_data_dir=str(raw_dir),
            processed_dir=str(processed_dir),
        )

        written = ingest(cfg)

        parquet_path = processed_dir / USER_DAY_FILENAME
        assert parquet_path.is_file(), f"ingest did not write {parquet_path}"

        read_back = pd.read_parquet(parquet_path, engine="pyarrow")

        # The written frame already carries a clean RangeIndex; read_parquet
        # produces the same, so a plain equality check covers rows, column
        # order, and dtypes.
        assert_frame_equal(
            read_back,
            written.reset_index(drop=True),
            check_like=False,
        )


if __name__ == "__main__":  # pragma: no cover - manual convenience entry point
    raise SystemExit(pytest.main([__file__, "-q"]))
