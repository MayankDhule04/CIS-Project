"""Single-command evaluation runner (Requirement 7.6).

Run this from the repository root with no extra setup::

    python scripts/run_eval.py

It puts ``src/`` on the import path, then calls the Evaluation_Module's
:func:`run_evaluation`, which reads ``risk.parquet`` (building it from the
synthetic pipeline if it is absent), compares the z-score-only, IsolationForest-
only, and fused variants, and writes ``reports/eval_report.md`` plus its plots.

Optional flags let you point it at a different config or output location (handy
for a throwaway run that must not overwrite the committed ``reports/``):

    python scripts/run_eval.py --out reports/eval_report.md --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Put the repository root and src/ on the path so ``import ueba`` works whether
# this script is launched from the repo root or elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from ueba.config import load_config  # noqa: E402
from ueba.evaluation import run_evaluation  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    """Parse arguments and run the evaluation end to end."""
    parser = argparse.ArgumentParser(
        description="Generate the UEBA evaluation report and plots."
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml."
    )
    parser.add_argument(
        "--out",
        default="reports/eval_report.md",
        help="Where to write the Markdown report.",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory for the plots (defaults to the report's directory).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild risk.parquet even if it already exists.",
    )
    parser.add_argument(
        "--data-source",
        choices=("synthetic", "real"),
        default="synthetic",
        help="Which dataset this run uses; selects the report Limitations text.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_evaluation(
        config=config,
        out_path=args.out,
        plots_dir=args.plots_dir,
        rebuild=args.rebuild,
        data_source=args.data_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
