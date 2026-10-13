"""Generate golden JSON from question definitions.

Usage:
    uv run python evals/build_golden.py

Reads:
    evals/fixtures/<name>.csv
    evals/questions/<name>.py

Writes:
    evals/golden/<name>.json

Run this whenever a question definition or the CSV changes. The consistency
test (tests/test_eval_titanic.py) verifies the generated JSON is up to date.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd

# Make `evals` package importable when this script runs standalone.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = REPO_ROOT / "evals" / "fixtures"
QUESTIONS_DIR = REPO_ROOT / "evals" / "questions"
GOLDEN_DIR = REPO_ROOT / "evals" / "golden"


def build_one(name: str, description: str = "") -> Path:
    """Build evals/golden/<name>.json from questions/<name>.py + fixtures/<name>.csv."""
    csv_path = FIXTURES_DIR / f"{name}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing fixture: {csv_path}")

    module = importlib.import_module(f"evals.questions.{name}")
    if not hasattr(module, "QUESTIONS"):
        raise AttributeError(f"evals.questions.{name} has no QUESTIONS list")

    df = pd.read_csv(csv_path)
    questions = []
    for q in module.QUESTIONS:
        expected = q["reference"](df)
        entry = {
            "id": q["id"],
            "question": q["question"],
            "difficulty": q["difficulty"],
            "expected_kind": q["expected_kind"],
            "expected": expected,
            "rel_tolerance": q.get("rel_tolerance", 0.01),
            "abs_tolerance": q.get("abs_tolerance", 1e-9),
        }
        if "notes" in q:
            entry["notes"] = q["notes"]
        questions.append(entry)

    payload = {
        "name": name,
        "csv_filename": f"{name}.csv",
        "description": description or f"Golden questions for {name}.",
        "questions": questions,
    }

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GOLDEN_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def main() -> None:
    targets = [
        ("sales", "Small e-commerce sales dataset for eval baseline."),
        ("titanic", "Classic Titanic passenger dataset."),
        ("ecommerce", "E-commerce orders with dates for time-series eval."),
    ]
    for name, desc in targets:
        path = build_one(name, desc)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
