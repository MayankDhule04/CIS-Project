"""Ingestion example and integration tests (task 2.7).

Two tests live here, each tied to specific acceptance criteria:

* ``test_ingestion_uses_chunked_reader`` — an *example* test proving the
  chunked-reader path is actually taken (Requirement 2.1). It wraps pandas'
  ``read_csv`` as the pipeline uses it and asserts every raw event file is read
  with a ``chunksize`` argument and that at least one file is iterated over in
  more than one chunk, which only happens when the file is larger than a single
  chunk. A whole-file load would never pass a ``chunksize`` nor iterate chunks.

* ``test_generator_to_ingestion_produces_valid_user_day`` — an *integration*
  test running the real path end to end: the Synthetic_Generator writes CERT
  4.2-shaped CSVs into a temp raw directory (Requirement 3.3, data usable
  downstream without the raw dataset), the ingestion pipeline consumes them
  (Requirement 2.7, passes on synthetic data), and the resulting
  ``user_day.parquet`` is validated for the invariants the pipeline promises.

Both tests use ``tempfile.TemporaryDirectory`` for the raw and processed dirs
and ``dataclasses.replace`` to repoint the loaded config at those temp dirs, so
nothing touches the repository's ``data/`` tree. This file deliberately defines
no fixtures and does not modify any shared conftest.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path

import pandas as pd

from ueba.config import load_config
from ueba.ingestion import pipeline
from ueba.ingestion.pipeline import DEFAULT_CHUNK_SIZE, USER_DAY_FILENAME, ingest
from ueba.ingestion.subset import ingest_subset

from generate_synthetic import SyntheticGenerator

# Repository root -> the project config.yaml, passed explicitly so these tests
# do not depend on the process working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "config.yaml"

# The raw event files the pipeline reads in chunks (their on-disk names).
_EVENT_FILENAMES = {"logon.csv", "device.csv", "http.csv", "email.csv"}

# Columns the finalised user_day table must carry.
_EXPECTED_USER_DAY_COLUMNS = {
    "user",
    "day",
    "role",
    "logon_events",
    "device_events",
    "http_events",
    "email_events",
    "total_events",
}


def _config_for(raw_dir: str, processed_dir: str):
    """Load the project config and repoint its data dirs at temp locations."""
    base = load_config(str(_CONFIG_PATH))
    return dataclasses.replace(
        base, raw_data_dir=raw_dir, processed_dir=processed_dir
    )


def test_ingestion_uses_chunked_reader():
    """Example: ingestion reads raw files in chunks, not as a whole-file load.

    Validates Requirement 2.1. A small ``chunk_size`` over a synthetic dataset
    big enough to exceed one chunk forces the chunked-reader path; we assert
    (a) every raw event file is opened with a ``chunksize`` argument equal to
    our chunk size, and (b) at least one file is iterated across more than one
    chunk, which is impossible for a single whole-file read.
    """
    chunk_size = 50

    with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as processed_dir:
        gen = SyntheticGenerator(config_path=str(_CONFIG_PATH))
        # Enough users/days that a 50-row chunk cannot hold a whole event file.
        gen.generate(raw_dir, n_users=5, n_days=40, seed=11)

        config = _config_for(raw_dir, processed_dir)

        # Wrap read_csv exactly as the pipeline module resolves it, recording
        # the chunksize each call used and how many chunks each reader yielded.
        real_read_csv = pipeline.pd.read_csv
        calls: list[dict] = []

        def recording_read_csv(filepath_or_buffer, *args, **kwargs):
            chunksize = kwargs.get("chunksize")
            record = {
                "name": os.path.basename(str(filepath_or_buffer)),
                "chunksize": chunksize,
                "chunks": 0,
            }
            calls.append(record)
            result = real_read_csv(filepath_or_buffer, *args, **kwargs)
            if chunksize is None:
                return result

            # Wrap the chunk iterator so we can count how many chunks the
            # pipeline actually pulls through, preserving real behaviour.
            def counting(reader, rec):
                for chunk in reader:
                    rec["chunks"] += 1
                    yield chunk

            return counting(result, record)

        original = pipeline.pd.read_csv
        pipeline.pd.read_csv = recording_read_csv
        try:
            result = ingest(config, chunk_size=chunk_size)
        finally:
            pipeline.pd.read_csv = original

        # Sanity: ingestion produced some user-day rows to work with.
        assert not result.empty

        # Every raw event file that was read must have been read with a
        # chunksize equal to the chunk size we requested (chunked path taken).
        event_calls = [c for c in calls if c["name"] in _EVENT_FILENAMES]
        assert event_calls, "no raw event files were read"
        for call in event_calls:
            assert call["chunksize"] == chunk_size, (
                f"{call['name']} was read without the expected chunksize "
                f"(got {call['chunksize']})"
            )

        # At least one event file must span more than one chunk; otherwise the
        # file fit in a single read and the chunking is not exercised.
        assert any(call["chunks"] > 1 for call in event_calls), (
            "no event file was iterated across multiple chunks; "
            f"chunk counts were {[(c['name'], c['chunks']) for c in event_calls]}"
        )


def test_generator_to_ingestion_produces_valid_user_day():
    """Integration: generator -> ingestion yields a valid user_day.parquet.

    Validates Requirements 2.7 and 3.3. The Synthetic_Generator writes a small
    dataset, the subset ingestion pipeline consumes it, and the persisted
    ``user_day.parquet`` is checked for the pipeline's invariants: it exists,
    is non-empty, carries the expected columns, has no null users, and holds
    exactly one row per ``(user, day)`` pair.
    """
    with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as processed_dir:
        gen = SyntheticGenerator(config_path=str(_CONFIG_PATH))
        gen.generate(raw_dir, n_users=10, n_days=60, seed=7)
        # Plant one anomaly so the labelled-insider path through subset
        # selection is exercised on real generator output as well.
        gen.plant_anomaly("USER0003", 45, "usb_burst")

        config = _config_for(raw_dir, processed_dir)
        result = ingest_subset(config)

        # The artifact is written where the config points.
        out_path = Path(processed_dir) / USER_DAY_FILENAME
        assert out_path.is_file(), "ingestion did not write user_day.parquet"

        on_disk = pd.read_parquet(out_path)

        # Non-empty and carrying exactly the expected feature-skeleton columns.
        assert not on_disk.empty
        assert set(on_disk.columns) == _EXPECTED_USER_DAY_COLUMNS, (
            f"unexpected columns: {sorted(on_disk.columns)}"
        )

        # No null users survived ingestion (Requirement 2.5 enforced upstream).
        assert on_disk["user"].notna().all()
        assert (on_disk["user"].astype("string").str.strip() != "").all()

        # Exactly one row per (user, day): no duplicate pairs.
        pair_counts = on_disk.groupby(["user", "day"]).size()
        assert (pair_counts == 1).all(), "found duplicate (user, day) rows"

        # The returned frame matches what was persisted to disk.
        assert len(result) == len(on_disk)
        assert set(result.columns) == set(on_disk.columns)

        # total_events is the sum of the per-source event counts.
        source_cols = [
            "logon_events",
            "device_events",
            "http_events",
            "email_events",
        ]
        expected_total = on_disk[source_cols].sum(axis=1)
        assert (on_disk["total_events"] == expected_total).all()
