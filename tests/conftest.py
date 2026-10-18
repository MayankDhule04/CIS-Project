"""Shared pytest fixtures and import-path setup for the test suite.

The UEBA source lives under ``src/ueba`` and tooling scripts live under
``scripts/``. This file puts the repository root, ``src/``, and ``scripts/``
on ``sys.path`` so tests can import both ``src.ueba.*`` (the style the profiler
uses) and the top-level ``ueba`` package, plus the ``profile_data`` script,
without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# tests/ -> repository root is its parent.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"

for _path in (REPO_ROOT, SRC_DIR, SCRIPTS_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def config_path(repo_root: Path) -> Path:
    """Absolute path to the project's ``config.yaml``."""
    return repo_root / "config.yaml"
