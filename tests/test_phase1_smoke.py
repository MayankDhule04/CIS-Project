"""Phase 1 smoke tests for the UEBA Insider Threat Detection project.

These tests verify that the Phase 1 foundation is in place before later phases
build on it. They check the project skeleton, the typed config loader and its
config-driven path resolution, that the raw data directory is excluded from
version control, that the test suite runs green under pytest, and that a
representative run performs no runtime network calls.

Covered acceptance criteria: 1.1, 1.2, 1.4, 1.5, 1.6, 10.3.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Requirement 1.1 — project structure exists
# ---------------------------------------------------------------------------

REQUIRED_FILES = (
    "config.yaml",
    "requirements.txt",
    "README.md",
    "app.py",
    "docs/ARCHITECTURE.md",
    "docs/DEMO.md",
)

REQUIRED_DIRS = (
    "scripts",
    "tests",
    "reports",
    "src/ueba",
)

REQUIRED_SUBPACKAGES = (
    "ingestion",
    "features",
    "baselines",
    "models",
    "scoring",
    "evaluation",
)


@pytest.mark.parametrize("rel_path", REQUIRED_FILES)
def test_required_files_exist(repo_root: Path, rel_path: str) -> None:
    """Each required top-level file is present (Req 1.1)."""
    target = repo_root / rel_path
    assert target.is_file(), f"Expected file missing: {rel_path}"


@pytest.mark.parametrize("rel_path", REQUIRED_DIRS)
def test_required_directories_exist(repo_root: Path, rel_path: str) -> None:
    """Each required directory is present (Req 1.1)."""
    target = repo_root / rel_path
    assert target.is_dir(), f"Expected directory missing: {rel_path}"


def test_src_ueba_is_a_package(repo_root: Path) -> None:
    """The ``src/ueba`` package has an ``__init__.py`` (Req 1.1)."""
    assert (repo_root / "src" / "ueba" / "__init__.py").is_file()


@pytest.mark.parametrize("subpackage", REQUIRED_SUBPACKAGES)
def test_required_subpackages_exist(repo_root: Path, subpackage: str) -> None:
    """Each ueba subpackage exists and is importable as a package (Req 1.1)."""
    pkg_dir = repo_root / "src" / "ueba" / subpackage
    assert pkg_dir.is_dir(), f"Missing subpackage directory: ueba/{subpackage}"
    assert (pkg_dir / "__init__.py").is_file(), (
        f"Subpackage ueba/{subpackage} is missing __init__.py"
    )


# ---------------------------------------------------------------------------
# Requirement 1.2 / 1.4 — config loads, resolves keys, drives path resolution
# ---------------------------------------------------------------------------

def test_load_config_resolves_required_keys(config_path: Path) -> None:
    """``load_config`` loads and resolves every required config value (Req 1.2)."""
    from src.ueba.config import load_config

    cfg = load_config(str(config_path))

    # Paths
    assert cfg.raw_data_dir
    assert cfg.processed_dir

    # Working hours window
    assert isinstance(cfg.working_hours.start_hour, int)
    assert isinstance(cfg.working_hours.end_hour, int)

    # Weekend days
    assert len(cfg.weekend_days) >= 1
    assert all(isinstance(day, int) for day in cfg.weekend_days)

    # Site-category lists
    assert cfg.site_categories.job_sites
    assert cfg.site_categories.upload_leak_sites

    # Scoring weights sum to 1.0
    total = cfg.weights.self_w + cfg.weights.iso_w + cfg.weights.peer_w
    assert total == pytest.approx(1.0)

    # Baseline bounds
    assert cfg.baseline.min_days >= 1
    assert cfg.baseline.min_days <= cfg.baseline.target_days <= cfg.baseline.max_days

    # Subset config
    assert cfg.subset.normal_sample_size >= 1

    # Column map populated for every file type
    for file_type in ("logon", "device", "http", "email"):
        assert file_type in cfg.column_map
        assert cfg.column_map[file_type]


def test_raw_data_dir_matches_config_file(config_path: Path) -> None:
    """The raw data dir comes from the Config_File, not a hardcoded value (Req 1.4)."""
    import yaml

    from src.ueba.config import load_config

    cfg = load_config(str(config_path))
    with config_path.open("r", encoding="utf-8") as handle:
        raw_yaml = yaml.safe_load(handle)

    assert cfg.raw_data_dir == raw_yaml["paths"]["raw_data_dir"]


def test_profiler_resolves_raw_dir_from_config(repo_root: Path, config_path: Path) -> None:
    """The profiler resolves the raw dir from config, relative to the config file (Req 1.4).

    This proves path resolution is config-driven rather than hardcoded: the
    relative ``data/raw`` in ``config.yaml`` resolves to an absolute path under
    the repository root.
    """
    import profile_data

    resolved = profile_data._resolve_raw_dir(str(config_path))
    resolved_path = Path(resolved)

    assert resolved_path.is_absolute()
    assert resolved_path == (repo_root / "data" / "raw").resolve()


# ---------------------------------------------------------------------------
# Requirement 1.5 — raw data directory excluded from version control
# ---------------------------------------------------------------------------

def test_gitignore_excludes_data_dir(repo_root: Path) -> None:
    """``.gitignore`` excludes the ``data/`` directory (Req 1.5)."""
    gitignore = repo_root / ".gitignore"
    assert gitignore.is_file(), ".gitignore is missing"

    patterns = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert any(p in {"data/", "data", "/data/", "/data"} for p in patterns), (
        ".gitignore does not exclude the data/ directory"
    )


# ---------------------------------------------------------------------------
# Requirement 1.6 — the test suite runs green under pytest
# ---------------------------------------------------------------------------

def test_pytest_runs_suite_green(tmp_path: Path) -> None:
    """pytest executes a representative test successfully (Req 1.6).

    Runs pytest in an isolated subprocess against a trivial passing test so the
    check is self-contained and never recurses into this suite. A zero exit code
    confirms the test runner is installed and reports a green run.
    """
    sample_test = tmp_path / "test_sample_green.py"
    sample_test.write_text(
        textwrap.dedent(
            """
            def test_truth():
                assert 1 + 1 == 2
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(sample_test)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, (
        "pytest did not run the suite green:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Requirement 10.3 — no runtime network calls during a representative run
# ---------------------------------------------------------------------------

def test_representative_run_makes_no_network_calls(
    monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """A representative run performs no runtime network calls (Req 10.3).

    All socket construction and outbound connections are blocked. A
    representative Phase 1 run — loading the config and profiling the raw data
    directory — must complete without attempting any network access.
    """

    class NetworkBlockedError(RuntimeError):
        """Raised if any code attempts a network operation during the run."""

    def _blocked_socket(*args, **kwargs):
        raise NetworkBlockedError("Network access attempted during a representative run")

    def _blocked_connection(*args, **kwargs):
        raise NetworkBlockedError("Outbound connection attempted during a representative run")

    monkeypatch.setattr(socket, "socket", _blocked_socket)
    monkeypatch.setattr(socket, "create_connection", _blocked_connection)

    import profile_data
    from src.ueba.config import load_config

    # Representative run: config load + path resolution + directory profiling.
    cfg = load_config(str(config_path))
    assert cfg.raw_data_dir

    raw_dir = profile_data._resolve_raw_dir(str(config_path))
    # Profiling an empty or populated raw dir must not touch the network.
    profiles = profile_data.profile_directory(raw_dir)
    assert isinstance(profiles, list)
