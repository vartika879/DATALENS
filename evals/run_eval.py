"""CLI: run an eval against a golden dataset.

Usage:
    uv run python evals/run_eval.py sales
    uv run python evals/run_eval.py titanic --limit 5
    uv run python evals/run_eval.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make repo root importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from datalens.evals.loader import (  # noqa: E402
    FIXTURES_DIR,
    GOLDEN_DIR,
    load_golden_dataset,
)
from datalens.evals.report import QuestionResult  # noqa: E402
from datalens.evals.runner import (  # noqa: E402
    EvalRunner,
    format_summary,
    write_report,
)

REPORTS_DIR = _REPO_ROOT / "evals" / "reports"


def _progress(i: int, total: int, r: QuestionResult) -> None:
    mark = "✅" if r.matches else ("❌" if r.execution_status == "success" else "⚠️")
    print(f"[{i:>3}/{total}] {mark}  {r.question_id:<15}{r.latency_ms:>6.0f} ms  {r.question[:60]}")


def _run_one(name: str, limit: int | None = None) -> None:
    golden_path = GOLDEN_DIR / f"{name}.json"
    dataset = load_golden_dataset(golden_path)

    if limit is not None:
        dataset.questions = dataset.questions[:limit]

    csv_path = FIXTURES_DIR / dataset.csv_filename
    print(f"Running eval: {dataset.name} ({len(dataset.questions)} questions)")
    print(f"CSV: {csv_path}")
    print()

    runner = EvalRunner(dataset=dataset, csv_path=csv_path)
    report = runner.run_all(progress=_progress)

    print(format_summary(report))
    out = write_report(report, REPORTS_DIR)
    print(f"Report saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DataLens evals.")
    parser.add_argument("dataset", nargs="?", help="Dataset name (sales, titanic, ...)")
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions")
    args = parser.parse_args()

    if args.all:
        names = ["sales", "titanic", "ecommerce", "census"]
    elif args.dataset:
        names = [args.dataset]
    else:
        parser.error("Pass a dataset name or --all")

    for name in names:
        _run_one(name, limit=args.limit)


if __name__ == "__main__":
    main()
