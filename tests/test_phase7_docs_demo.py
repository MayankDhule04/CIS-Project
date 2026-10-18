"""Phase 7 smoke and integration tests (task 7.3).

Phase 7 is the documentation-and-demo-polish phase. There is no new runtime
behavior to exercise here; instead these tests verify two things the phase
promises:

* **Smoke (Req 9.1-9.4, 10.1):** the required documents exist and actually
  contain the sections the spec calls for (Mermaid diagrams, the feature table,
  the scoring formula and final weights, the evaluation results, and the
  limitations in ``ARCHITECTURE.md``; the demo walkthrough in ``DEMO.md``; the
  clone-and-run instructions and doc links in ``README.md``), every source
  module carries a plain-English module docstring, and the per-phase test files
  that cover the seven-phase plan are present.

* **Integration (Req 9.5):** a fresh clone that follows the README can build the
  synthetic data and launch the dashboard on it. We reproduce that path in a
  throwaway temp area — generate synthetic logs, run the single-command
  evaluation to build ``risk.parquet`` and the report — then confirm the real
  ``app.py`` renders the ranking on that synthetic data without raising.

Design choices worth knowing:

* The documentation checks are deliberately *robust* (case-insensitive
  substring / regex, "at least N of" thresholds) rather than brittle exact-text
  matches, so legitimate wording tweaks in the docs do not break the suite while
  a genuinely missing section still fails loudly.
* The integration run uses a temp ``config.yaml`` whose ``raw_data_dir`` and
  ``processed_dir`` point at temp directories, so generating data and building
  ``risk.parquet`` never touches the committed repo. The scripts are invoked as
  subprocesses exactly as a new contributor would run them.
* ``app.py`` loads the repo's own ``config.yaml`` (it cannot be redirected to the
  temp config), so to render the dashboard on the freshly built synthetic data we
  copy the temp-built ``risk.parquet`` into the repo's ``data/processed`` — after
  backing up any existing file — run the app headlessly, then restore the repo to
  exactly its prior state in a ``finally`` block. This mirrors
  ``tests/test_dashboard_integration.py``.

These tests do not modify any implementation or documentation.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Streamlit's headless app-testing harness (present in modern Streamlit).
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    """Read a UTF-8 text file from the repo."""
    return path.read_text(encoding="utf-8")


def _assert_contains(text: str, needle: str, where: str, *, ignore_case: bool = True) -> None:
    """Assert ``needle`` appears in ``text`` (case-insensitive by default)."""
    hay = text.lower() if ignore_case else text
    probe = needle.lower() if ignore_case else needle
    assert probe in hay, f"{where} is missing expected content: {needle!r}"


# The 15 behavioral feature names documented in ARCHITECTURE.md's feature table.
_FEATURE_NAMES = [
    "logon_count",
    "after_hours_logon_count",
    "weekend_logon_count",
    "distinct_machines",
    "usb_connect_count",
    "after_hours_usb_count",
    "files_to_removable",
    "job_site_visits",
    "upload_leak_site_visits",
    "file_upload_count",
    "distinct_domains",
    "total_web_requests",
    "emails_sent",
    "external_recipients",
    "email_attachments",
]

# Source modules that must each carry a plain-English module docstring (Req 9.4).
_MODULE_FILES = [
    "src/ueba/config.py",
    "src/ueba/ingestion/pipeline.py",
    "src/ueba/features/extractor.py",
    "src/ueba/baselines/builder.py",
    "src/ueba/models/isolation.py",
    "src/ueba/scoring/statistical.py",
    "src/ueba/scoring/engine.py",
    "src/ueba/evaluation/metrics.py",
    "src/ueba/evaluation/report.py",
    "src/ueba/dashboard/logic.py",
    "app.py",
    "scripts/profile_data.py",
    "scripts/generate_synthetic.py",
    "scripts/run_eval.py",
]

# A representative set of per-phase test files proving the seven-phase plan is
# covered by per-phase tests (Req 10.1, 10.2). Phase 1 -> Phase 6 here; Phase 7
# is this file.
_PER_PHASE_TEST_FILES = [
    # Phase 1 — setup & profiling
    "tests/test_phase1_smoke.py",
    "tests/test_property1_profiler.py",
    # Phase 2 — ingestion + generator
    "tests/test_property6_7_generator.py",
    "tests/test_property2_3_4_ingestion.py",
    "tests/test_property5_roundtrip_userday.py",
    "tests/test_ingestion_integration.py",
    # Phase 3 — features, baselines, statistical scoring
    "tests/test_property8_9_features.py",
    "tests/test_property10_11_baselines.py",
    "tests/test_property12_5_statistical.py",
    "tests/test_scoring_edge_cases.py",
    # Phase 4 — isolation + fusion
    "tests/test_property13_18_isolation.py",
    "tests/test_property14_17_fusion.py",
    # Phase 5 — evaluation
    "tests/test_property19_evaluation.py",
    "tests/test_evaluation_integration.py",
    # Phase 6 — dashboard
    "tests/test_property20_22_dashboard.py",
    "tests/test_dashboard_integration.py",
]


# ===========================================================================
# 1. Documentation smoke tests (Req 9.1, 9.2, 9.3)
# ===========================================================================

def test_required_docs_exist_and_are_nontrivial():
    """The three required documents exist and carry real content (Req 9.1-9.3)."""
    for rel in ("README.md", "docs/ARCHITECTURE.md", "docs/DEMO.md"):
        path = REPO_ROOT / rel
        assert path.is_file(), f"required document missing: {rel}"
        # "Non-trivial" — more than a placeholder stub.
        assert len(_read(path).strip()) > 500, f"{rel} looks like a stub (too short)"


def test_architecture_has_required_sections():
    """ARCHITECTURE.md contains every section Req 9.1 requires.

    Checks (robustly): at least one Mermaid diagram, the feature table with the
    documented feature names, the scoring formula, the final 0.5/0.3/0.2 weights,
    an evaluation-results section, and a limitations section.
    """
    text = _read(REPO_ROOT / "docs" / "ARCHITECTURE.md")
    low = text.lower()

    # --- Mermaid diagrams ---
    mermaid_fences = len(re.findall(r"```mermaid", text, flags=re.IGNORECASE))
    assert mermaid_fences >= 1, "ARCHITECTURE.md has no ```mermaid diagram fence"

    # --- Feature table ---
    # A Markdown table header mentioning "Feature", plus most of the 15 names.
    assert re.search(r"\|\s*feature\s*\|", low), "no feature table header found"
    present = [name for name in _FEATURE_NAMES if name.lower() in low]
    assert len(present) >= 13, (
        "feature table is missing feature names; "
        f"found only {len(present)}/15: {present}"
    )

    # --- Scoring formula ---
    # daily_risk = 100 * (weighted sum), plus the weekly-risk definition.
    assert re.search(r"daily_risk\s*=\s*100\s*\*", low), (
        "scoring formula (daily_risk = 100 * ...) not found"
    )
    _assert_contains(text, "weekly_risk", "ARCHITECTURE.md scoring formula")

    # --- Final weights (self 0.5, iso 0.3, peer 0.2) ---
    for value in ("0.5", "0.3", "0.2"):
        assert value in text, f"final weight value {value} not documented"
    for label in ("self", "iso", "peer"):
        assert label in low, f"weight label {label!r} not documented"

    # --- Evaluation results ---
    assert "evaluation" in low, "no evaluation section"
    assert "recall@20" in low, "evaluation section lacks recall@20"
    assert "pr-auc" in low, "evaluation section lacks PR-AUC"

    # --- Limitations ---
    assert "limitation" in low, "no limitations section"


def test_demo_has_steps_and_prep_commands():
    """DEMO.md describes the demo steps and the prep commands (Req 9.2)."""
    text = _read(REPO_ROOT / "docs" / "DEMO.md")
    low = text.lower()

    # Demo narrative touchpoints.
    for topic in ("dashboard", "threshold", "evaluation", "limitation"):
        assert topic in low, f"DEMO.md does not mention {topic!r}"
    # Why-flagged / reasons explainability story (either phrasing).
    assert ("why-flagged" in low or "why flagged" in low or "reason" in low), (
        "DEMO.md does not cover the why-flagged / reasons panel"
    )

    # Prep commands a presenter runs before the demo.
    assert "generate_synthetic" in low, "DEMO.md missing generate_synthetic step"
    assert "run_eval" in low, "DEMO.md missing run_eval step"
    assert "streamlit run app.py" in low, "DEMO.md missing 'streamlit run app.py'"


def test_readme_has_setup_quickstart_tests_and_links():
    """README.md has setup, the synthetic quickstart, tests, and doc links (Req 9.3)."""
    text = _read(REPO_ROOT / "README.md")
    low = text.lower()

    # Setup.
    assert "pip install -r requirements.txt" in low, "README missing dependency install step"

    # Synthetic quickstart commands.
    assert "generate_synthetic" in low, "README missing generate_synthetic quickstart"
    assert "run_eval" in low, "README missing run_eval quickstart"
    assert "streamlit run app.py" in low, "README missing 'streamlit run app.py'"

    # Tests command.
    assert "pytest" in low, "README missing a pytest command"

    # Links to the companion docs.
    assert "architecture.md" in low, "README does not link to ARCHITECTURE.md"
    assert "demo.md" in low, "README does not link to DEMO.md"


# ===========================================================================
# 2. Per-module explanation smoke (Req 9.4)
# ===========================================================================

@pytest.mark.parametrize("rel_path", _MODULE_FILES)
def test_module_has_plain_english_docstring(rel_path):
    """Every source module / script begins with a substantial module docstring."""
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"expected module file missing: {rel_path}"

    tree = ast.parse(_read(path), filename=str(path))
    docstring = ast.get_docstring(tree)

    assert docstring is not None, f"{rel_path} has no module-level docstring"
    # A plain-English explanation, not a one-word placeholder.
    assert len(docstring.strip()) >= 40, (
        f"{rel_path} module docstring is too short to be a real explanation"
    )


# ===========================================================================
# 3. Phase-plan / per-phase tests exist (Req 10.1, 10.2)
# ===========================================================================

@pytest.mark.parametrize("rel_path", _PER_PHASE_TEST_FILES)
def test_per_phase_test_file_exists(rel_path):
    """The per-phase test files covering the seven-phase plan are present."""
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"per-phase test file missing: {rel_path}"


def test_phase_plan_is_covered_by_tests():
    """Each of phases 1-6 has at least one dedicated test file, and phase 7 is here.

    Proves the plan's per-phase testing (Req 10.1/10.2) is realized: we group the
    representative test files by the phase they cover and require every phase to
    have at least one file present.
    """
    phase_markers = {
        "phase1": ["test_phase1_smoke.py", "test_property1_profiler.py"],
        "phase2": [
            "test_property6_7_generator.py",
            "test_property2_3_4_ingestion.py",
            "test_ingestion_integration.py",
        ],
        "phase3": [
            "test_property8_9_features.py",
            "test_property10_11_baselines.py",
            "test_property12_5_statistical.py",
        ],
        "phase4": ["test_property13_18_isolation.py", "test_property14_17_fusion.py"],
        "phase5": ["test_property19_evaluation.py", "test_evaluation_integration.py"],
        "phase6": ["test_property20_22_dashboard.py", "test_dashboard_integration.py"],
    }
    tests_dir = REPO_ROOT / "tests"
    for phase, files in phase_markers.items():
        present = [f for f in files if (tests_dir / f).is_file()]
        assert present, f"{phase} has no per-phase test file present"

    # Phase 7 (documentation & demo) is covered by this very file.
    assert (tests_dir / "test_phase7_docs_demo.py").is_file()


# ===========================================================================
# 4. Fresh-clone demo integration (Req 9.5)
# ===========================================================================

def _write_temp_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Copy the repo config.yaml, redirecting raw/processed dirs into ``tmp_path``.

    Returns ``(temp_config_path, temp_raw_dir, temp_processed_dir)``. Paths are
    written POSIX-style so YAML stays clean on Windows.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(_read(REPO_ROOT / "config.yaml"))
    config["paths"]["raw_data_dir"] = raw_dir.as_posix()
    config["paths"]["processed_dir"] = processed_dir.as_posix()

    temp_config = tmp_path / "config.yaml"
    temp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return temp_config, raw_dir, processed_dir


def _run_script(args: list[str]) -> subprocess.CompletedProcess:
    """Run a repo script as a subprocess with the repo + src on PYTHONPATH."""
    env = dict(os.environ)
    pythonpath = os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = pythonpath + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_fresh_clone_builds_synthetic_data_and_launches_dashboard(tmp_path):
    """Req 9.5: following the README path builds synthetic data and the dashboard renders.

    1. Generate CERT-shaped synthetic logs into a temp ``raw`` dir (with a planted
       insider) via ``scripts/generate_synthetic.py``.
    2. Build ``risk.parquet`` + the evaluation report into temp dirs via the
       single-command ``scripts/run_eval.py --rebuild``.
    3. Confirm both subprocesses exit 0 and the temp ``risk.parquet`` and
       ``eval_report.md`` were created.
    4. Copy the temp-built ``risk.parquet`` into the repo's configured
       ``data/processed`` (backing up any existing file), run the real ``app.py``
       headlessly, assert it renders the ranking without raising, then restore the
       repo exactly.
    """
    temp_config, temp_raw, temp_processed = _write_temp_config(tmp_path)

    # --- Step 1: generate synthetic logs with a planted insider ---
    gen = _run_script(
        [
            "scripts/generate_synthetic.py",
            "--config", str(temp_config),
            "--out-dir", str(temp_raw),
            "--n-users", "30",
            "--n-days", "90",
            "--seed", "42",
            "--plant", "USER0005:80:usb_burst",
        ]
    )
    assert gen.returncode == 0, (
        "generate_synthetic.py failed\n"
        f"STDOUT:\n{gen.stdout}\nSTDERR:\n{gen.stderr}"
    )
    # Synthetic event CSVs landed in the temp raw dir.
    assert (temp_raw / "logon.csv").is_file(), "generator did not write logon.csv"

    # --- Step 2: single-command build of risk.parquet + report ---
    temp_report = tmp_path / "eval_report.md"
    evaluate = _run_script(
        [
            "scripts/run_eval.py",
            "--config", str(temp_config),
            "--out", str(temp_report),
            "--plots-dir", str(tmp_path),
            "--rebuild",
        ]
    )
    assert evaluate.returncode == 0, (
        "run_eval.py failed\n"
        f"STDOUT:\n{evaluate.stdout}\nSTDERR:\n{evaluate.stderr}"
    )

    # --- Step 3: artifacts exist in the temp area (repo untouched so far) ---
    temp_risk = temp_processed / "risk.parquet"
    assert temp_risk.is_file(), "run_eval.py did not build risk.parquet in the temp dir"
    assert temp_report.is_file(), "run_eval.py did not write the evaluation report"

    # --- Step 4: launch the real dashboard on that synthetic risk.parquet ---
    # app.py loads the repo's config.yaml, so point the repo's processed dir at
    # the freshly built file, backing up and restoring the original.
    repo_processed = REPO_ROOT / "data" / "processed"
    repo_processed.mkdir(parents=True, exist_ok=True)
    repo_risk = repo_processed / "risk.parquet"
    backup = repo_risk.with_name("risk.parquet.bak_phase7") if repo_risk.exists() else None

    try:
        if backup is not None:
            shutil.copy2(repo_risk, backup)
        shutil.copy2(temp_risk, repo_risk)

        # Clear Streamlit's data cache so the app reads the file we just wrote
        # rather than a value cached by an earlier dashboard test.
        import streamlit as st

        st.cache_data.clear()

        app_path = str(REPO_ROOT / "app.py")
        app = AppTest.from_file(app_path, default_timeout=120)
        app.run()

        assert not app.exception, (
            f"dashboard raised on synthetic data: {[str(e) for e in app.exception]}"
        )
        # The ranking rendered: at least one dataframe (the ranked-users table).
        assert len(app.dataframe) >= 1, "dashboard did not render the ranking table"
    finally:
        if backup is not None:
            shutil.move(str(backup), str(repo_risk))
        elif repo_risk.exists():
            repo_risk.unlink()
