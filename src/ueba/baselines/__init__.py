"""Baselines layer: per-user self and peer statistical baselines.

Exposes the Baseline_Builder: it picks each user's early-history
Baseline_Period (first 60–90 days, minimum 30), then summarises each
behavioural feature as a mean and standard deviation for the user (self
baseline) and for their job-role group (peer baseline). Baselines are built
from in-period user-days only; short-history users are flagged insufficient so
scoring can skip them, and zero-variance features get a standard deviation of
``0`` so the downstream z-score is defined as ``0`` without dividing by zero.
See :mod:`src.ueba.baselines.builder`.
"""

from .builder import (
    Baseline,
    baseline_cutoff,
    build_baselines,
    build_peer_baseline,
    build_self_baseline,
    select_baseline_rows,
    split_by_cutoff,
)

__all__ = [
    "Baseline",
    "baseline_cutoff",
    "build_baselines",
    "build_peer_baseline",
    "build_self_baseline",
    "select_baseline_rows",
    "split_by_cutoff",
]
