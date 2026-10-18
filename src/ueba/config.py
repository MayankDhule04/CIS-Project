"""Typed configuration loader for the UEBA system.

This module is the single place the whole system reads its settings from.
It opens ``config.yaml`` once, checks that the important values make sense,
and hands back a read-only ``Config`` object. Every other module asks this
object for paths, the working-hours window, weekend days, site-category
lists, the scoring weights, baseline-window bounds, the subset size, and the
column map — so no path, column name, window, or weight is ever hard-coded
anywhere else.

Two guardrails run at load time so mistakes surface immediately rather than
deep inside the pipeline:

* The three fusion weights (self, isolation, peer) must add up to 1.0 within
  a small tolerance.
* Every file type in ``column_map`` must list its logical column names, so a
  parser never discovers a missing mapping halfway through a 20 GB read.

A bad config raises :class:`ConfigError` with a message that names exactly
what is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Fusion weights must sum to 1.0; allow a small floating-point tolerance.
WEIGHT_SUM_TOLERANCE = 1e-6

# Logical column names each file type must provide in column_map.
REQUIRED_COLUMN_MAP: dict[str, tuple[str, ...]] = {
    "logon": ("user", "date", "pc", "activity"),
    "device": ("user", "date", "pc", "activity"),
    "http": ("user", "date", "url"),
    "email": ("user", "date", "to", "attachments", "size"),
}


class ConfigError(Exception):
    """Raised when ``config.yaml`` is missing, malformed, or invalid."""


@dataclass(frozen=True)
class WorkingHours:
    """Working-hours window used to classify after-hours activity.

    ``start_hour`` is inclusive and ``end_hour`` is exclusive; any timestamp
    whose hour falls outside ``[start_hour, end_hour)`` is after-hours.
    """

    start_hour: int
    end_hour: int


@dataclass(frozen=True)
class SiteCategories:
    """Configured web-site category lists used by feature extraction."""

    job_sites: tuple[str, ...]
    upload_leak_sites: tuple[str, ...]


@dataclass(frozen=True)
class FusionWeights:
    """Weights that fuse the self, isolation, and peer scores into risk."""

    self_w: float
    iso_w: float
    peer_w: float


@dataclass(frozen=True)
class BaselineConfig:
    """Bounds for each user's baseline window, in days."""

    min_days: int
    target_days: int
    max_days: int


@dataclass(frozen=True)
class SubsetConfig:
    """Working-subset sampling settings (all insiders plus sampled normals)."""

    normal_sample_size: int
    seed: int


@dataclass(frozen=True)
class Config:
    """Read-only typed view of ``config.yaml``.

    This is the single access point for every configurable value in the
    system. It is frozen so no module can mutate shared settings at runtime.
    """

    raw_data_dir: str
    processed_dir: str
    working_hours: WorkingHours
    weekend_days: tuple[int, ...]
    site_categories: SiteCategories
    weights: FusionWeights
    baseline: BaselineConfig
    subset: SubsetConfig
    column_map: dict[str, dict[str, str]]


def _require(mapping: Any, key: str, context: str) -> Any:
    """Return ``mapping[key]`` or raise a clear :class:`ConfigError`."""
    if not isinstance(mapping, dict):
        raise ConfigError(f"Expected a mapping for '{context}', got {type(mapping).__name__}.")
    if key not in mapping:
        raise ConfigError(f"Missing required config key '{key}' under '{context}'.")
    return mapping[key]


def _validate_weights(weights: FusionWeights) -> None:
    """Reject fusion weights that do not sum to 1.0 within tolerance."""
    total = weights.self_w + weights.iso_w + weights.peer_w
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ConfigError(
            "Fusion weights must sum to 1.0 "
            f"(self_w={weights.self_w}, iso_w={weights.iso_w}, "
            f"peer_w={weights.peer_w}; sum={total})."
        )


def _validate_column_map(column_map: dict[str, dict[str, str]]) -> None:
    """Ensure every file type lists all of its required logical names."""
    for file_type, required_names in REQUIRED_COLUMN_MAP.items():
        if file_type not in column_map:
            raise ConfigError(
                f"column_map is missing an entry for the '{file_type}' file type."
            )
        mapping = column_map[file_type]
        if not isinstance(mapping, dict):
            raise ConfigError(
                f"column_map entry for '{file_type}' must be a mapping of "
                f"logical name -> source column."
            )
        for logical_name in required_names:
            if logical_name not in mapping:
                raise ConfigError(
                    f"column_map for file type '{file_type}' is missing the "
                    f"logical field '{logical_name}'."
                )


def load_config(path: str = "config.yaml") -> Config:
    """Load, validate, and return the typed :class:`Config`.

    Reads the raw and processed data directories, the working-hours window,
    weekend days, site-category lists, fusion weights, baseline bounds, the
    subset config, and the column map from ``path``. Raises
    :class:`ConfigError` if the file is missing or malformed, if the fusion
    weights do not sum to 1.0, or if any ``column_map`` logical name is absent.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config root in {config_path} must be a mapping.")

    paths = _require(raw, "paths", "config")
    raw_data_dir = _require(paths, "raw_data_dir", "paths")
    processed_dir = _require(paths, "processed_dir", "paths")

    wh = _require(raw, "working_hours", "config")
    working_hours = WorkingHours(
        start_hour=int(_require(wh, "start_hour", "working_hours")),
        end_hour=int(_require(wh, "end_hour", "working_hours")),
    )

    weekend_days = tuple(int(day) for day in _require(raw, "weekend_days", "config"))

    sc = _require(raw, "site_categories", "config")
    site_categories = SiteCategories(
        job_sites=tuple(_require(sc, "job_sites", "site_categories")),
        upload_leak_sites=tuple(_require(sc, "upload_leak_sites", "site_categories")),
    )

    w = _require(raw, "weights", "config")
    weights = FusionWeights(
        self_w=float(_require(w, "self_w", "weights")),
        iso_w=float(_require(w, "iso_w", "weights")),
        peer_w=float(_require(w, "peer_w", "weights")),
    )
    _validate_weights(weights)

    b = _require(raw, "baseline", "config")
    baseline = BaselineConfig(
        min_days=int(_require(b, "min_days", "baseline")),
        target_days=int(_require(b, "target_days", "baseline")),
        max_days=int(_require(b, "max_days", "baseline")),
    )

    s = _require(raw, "subset", "config")
    subset = SubsetConfig(
        normal_sample_size=int(_require(s, "normal_sample_size", "subset")),
        seed=int(_require(s, "seed", "subset")),
    )

    raw_column_map = _require(raw, "column_map", "config")
    if not isinstance(raw_column_map, dict):
        raise ConfigError("column_map must be a mapping of file type -> column mapping.")
    column_map = {
        file_type: dict(mapping) for file_type, mapping in raw_column_map.items()
    }
    _validate_column_map(column_map)

    return Config(
        raw_data_dir=str(raw_data_dir),
        processed_dir=str(processed_dir),
        working_hours=working_hours,
        weekend_days=weekend_days,
        site_categories=site_categories,
        weights=weights,
        baseline=baseline,
        subset=subset,
        column_map=column_map,
    )
