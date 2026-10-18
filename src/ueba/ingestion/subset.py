"""Subset selection — pick the working slice of users to actually process.

The full CERT 4.2 dataset has far more users than a demo needs, and the vast
majority of them are ordinary employees who never do anything suspicious.
Processing every one of them is wasteful. Instead the system works on a
*Subset* (Requirement 2.6): **every** labeled insider user, plus a modest,
reproducibly-sampled handful of normal (non-insider) users to serve as a
realistic backdrop.

Two pieces live here:

* :func:`select_subset` — the pure, deterministic core. Given the full user
  population, the set of known insiders, a sample size, and a seed, it returns
  all insiders plus up to ``sample_size`` randomly chosen normals. Same inputs
  and same seed always yield the same subset, so a run is reproducible.
* :func:`ingest_subset` — the convenience wrapper that wires that core into
  the ingestion pipeline. It reads the ground-truth insiders from
  ``labels.csv``, works out the full user population from ``roles.csv`` (or the
  raw event files as a fallback), calls :func:`select_subset` with the
  subset size and seed from ``config``, and hands the chosen users to
  :func:`~src.ueba.ingestion.pipeline.ingest` via its ``users`` hook.

Keeping the sampling logic separate from :func:`ingest` means ``ingest`` stays
a general tool (restrict output to *any* given set of users) while the policy
of *which* users form the subset lives in one small, testable place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..config import Config
from .pipeline import EVENT_FILES, ingest

# Ground-truth labels file written by the Synthetic_Generator (and shipped
# with the real dataset): columns user, day, insider.
LABELS_FILENAME = "labels.csv"

# User -> role lookup, also used to enumerate the full user population.
ROLES_FILENAME = "roles.csv"


def select_subset(
    users: Iterable[str],
    insiders: Iterable[str],
    sample_size: int,
    seed: int,
) -> set[str]:
    """Return all insiders plus a reproducible sample of normal users.

    This is the deterministic heart of subset selection. It always keeps every
    insider that appears in ``users`` and then draws up to ``sample_size`` of
    the remaining (normal) users using a seeded random generator, so the same
    inputs and seed always produce the same subset.

    Args:
        users: The full population of user ids to choose from. Duplicates and
            blank/null ids are ignored.
        insiders: The known insider user ids. Every insider that is present in
            ``users`` is always included in the result. Insiders that are not
            in ``users`` are still included, so no known insider is ever lost.
        sample_size: The approximate number of *normal* (non-insider) users to
            add. If fewer normals exist than ``sample_size``, all of them are
            included. Values below zero are treated as zero.
        seed: Seed for the random sample, making selection reproducible.

    Returns:
        A set of user ids: every insider plus up to ``sample_size`` sampled
        normal users.
    """
    # Normalise inputs to clean, de-duplicated string sets.
    population = _clean_id_set(users)
    insider_set = _clean_id_set(insiders)

    # Insiders are always kept. Keep known insiders even if they somehow are
    # not listed in the population, so a labeled insider is never dropped.
    selected: set[str] = set(insider_set)

    # Normals are the population members that are not insiders.
    normals = sorted(population - insider_set)

    target = max(0, int(sample_size))
    if target >= len(normals):
        # Not enough normals to sample from — take them all.
        selected.update(normals)
        return selected

    # Draw a reproducible sample without replacement. We sort first and use a
    # seeded generator so the choice depends only on the inputs and the seed,
    # never on dict/set iteration order.
    import numpy as np

    rng = np.random.default_rng(seed)
    chosen_index = rng.choice(len(normals), size=target, replace=False)
    selected.update(normals[i] for i in chosen_index)
    return selected


def read_insiders(raw_dir: Path) -> set[str]:
    """Read the set of known insider user ids from ``labels.csv``.

    The labels file lists ground-truth insider user-days (columns ``user``,
    ``day``, ``insider``). A user counts as an insider if they have any row
    whose ``insider`` flag is truthy. A missing or empty labels file simply
    yields an empty set (no insiders known), which is a valid situation.
    """
    labels_path = raw_dir / LABELS_FILENAME
    if not labels_path.is_file():
        return set()
    labels = pd.read_csv(labels_path, dtype=str)
    if "user" not in labels.columns:
        return set()
    if "insider" in labels.columns:
        flag = pd.to_numeric(labels["insider"], errors="coerce").fillna(0)
        labels = labels[flag > 0]
    return _clean_id_set(labels["user"])


def read_population(raw_dir: Path) -> set[str]:
    """Determine the full user population for the raw data directory.

    Prefers ``roles.csv`` (one row per user) because it is small and explicit.
    If roles are unavailable, falls back to scanning the raw event CSVs for
    their user columns. The result is every distinct, non-blank user id found.
    """
    roles_path = raw_dir / ROLES_FILENAME
    if roles_path.is_file():
        roles = pd.read_csv(roles_path, dtype=str)
        if "user" in roles.columns:
            population = _clean_id_set(roles["user"])
            if population:
                return population

    # Fallback: gather users from the event files. Each generator file names
    # its user column "user"; read only that column to stay light.
    population: set[str] = set()
    for file_type in EVENT_FILES:
        file_path = raw_dir / f"{file_type}.csv"
        if not file_path.is_file():
            continue
        try:
            column = pd.read_csv(file_path, usecols=["user"], dtype=str)
        except (ValueError, KeyError):
            # "user" column absent in this file; skip it.
            continue
        population |= _clean_id_set(column["user"])
    return population


def ingest_subset(
    config: Config,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Ingest only the Subset: all insiders plus sampled normal users.

    This is the convenience entry point for Requirement 2.6. It figures out the
    subset membership from the data on disk and the ``subset`` settings in
    ``config``, then runs the normal ingestion pipeline restricted to those
    users.

    Steps, in plain English:

    1. Read the ground-truth insiders from ``labels.csv``.
    2. Work out the full user population from ``roles.csv`` (or the event files).
    3. Call :func:`select_subset` with ``config.subset.normal_sample_size`` and
       ``config.subset.seed`` to choose all insiders plus a reproducible sample
       of normals.
    4. Call :func:`~src.ueba.ingestion.pipeline.ingest` with that set via its
       ``users`` hook, so the written ``user_day.parquet`` holds only the
       Subset.

    Args:
        config: The loaded :class:`~src.ueba.config.Config`. Supplies the raw
            directory, the subset size, and the seed.
        chunk_size: Optional override for the ingestion chunk size; when
            ``None`` the pipeline's default is used.

    Returns:
        The per-user-day DataFrame restricted to the Subset (also written to
        ``user_day.parquet`` by the pipeline).
    """
    raw_dir = Path(config.raw_data_dir)

    insiders = read_insiders(raw_dir)
    population = read_population(raw_dir)

    subset_users = select_subset(
        users=population,
        insiders=insiders,
        sample_size=config.subset.normal_sample_size,
        seed=config.subset.seed,
    )

    if chunk_size is None:
        return ingest(config, users=subset_users)
    return ingest(config, users=subset_users, chunk_size=chunk_size)


def _clean_id_set(values: Iterable[str]) -> set[str]:
    """Return a set of trimmed, non-empty string ids from any iterable/series.

    Accepts a pandas Series or any iterable. Null/NaN values and ids that are
    blank once whitespace is stripped are discarded, so callers never have to
    worry about stray empty users sneaking into the subset.
    """
    series = pd.Series(list(values), dtype="object")
    series = series.astype("string").str.strip()
    series = series[series.notna() & (series != "")]
    return set(series.tolist())
