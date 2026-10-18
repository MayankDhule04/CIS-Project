"""Run the evaluation as a module: ``python -m ueba.evaluation``.

This is the package-level shortcut for the Evaluation_Module's single-command
entry point (Requirement 7.6). It simply calls :func:`run_evaluation`, which
builds or loads ``risk.parquet``, compares the three scoring variants, and
writes ``reports/eval_report.md`` and its plots.

It works when ``src/`` is on the import path (as the test suite arranges). From
a clean shell at the repository root, the equivalent one command that needs no
path setup is ``python scripts/run_eval.py``.
"""

from __future__ import annotations

from .report import run_evaluation

if __name__ == "__main__":
    run_evaluation()
