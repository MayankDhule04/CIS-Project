"""Real-data adapters for the CERT r4.2 dataset.

The synthetic generator ships the two helper files the pipeline needs —
``roles.csv`` (user -> role) and ``labels.csv`` (user, day, insider) — right
next to the event logs. The *real* CERT r4.2 dataset does not: roles live in a
tree of monthly LDAP snapshots, and the ground-truth insider labels ship
separately (an ``answers/`` tree and/or a published ``insiders.csv``).

This package bridges that gap. It converts the real dataset's own layout into
the exact two helper files every downstream stage already consumes, without
changing anything in ingestion, features, scoring, or evaluation.

Public entry points:

* :func:`build_roles_from_ldap` — collapse the monthly LDAP snapshots into a
  single ``roles.csv`` (columns ``user``, ``role``).
* :func:`build_labels_from_answers` — convert CERT answers (an ``answers/``
  directory or a published ``insiders.csv``) into ``labels.csv`` (columns
  ``user``, ``day``, ``insider``).
* :func:`build_labels_for_release` — the release-safe variant that labels from
  **one** CERT release's scenario directories only (e.g. ``r4.2``), so a
  multi-release ``answers/`` tree never leaks other releases' insiders.
* :func:`collect_release_scenario_dirs` — list the per-scenario answer
  directories belonging to a single release.
"""

from __future__ import annotations

from .adapters import (
    build_labels_for_release,
    build_labels_from_answers,
    build_roles_from_ldap,
    collect_release_scenario_dirs,
)

__all__ = [
    "build_roles_from_ldap",
    "build_labels_from_answers",
    "build_labels_for_release",
    "collect_release_scenario_dirs",
]
