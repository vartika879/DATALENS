"""Golden dataset loader.

Loads a GoldenDataset from a JSON file and validates that the CSV it
references exists (relative to evals/fixtures/).
"""

from __future__ import annotations

import json
from pathlib import Path

from datalens.evals.schema import GoldenDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVALS_DIR = REPO_ROOT / "evals"
FIXTURES_DIR = EVALS_DIR / "fixtures"
GOLDEN_DIR = EVALS_DIR / "golden"


class GoldenDatasetError(Exception):
    """Raised when a golden dataset cannot be loaded or is invalid."""


def load_golden_dataset(path: Path) -> GoldenDataset:
    """Load and validate a golden dataset from a JSON file.

    Raises:
        GoldenDatasetError: If the file is missing, malformed, or the
            referenced CSV does not exist.
    """
    if not path.is_file():
        raise GoldenDatasetError(f"Golden dataset not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GoldenDatasetError(f"Invalid JSON in {path}: {exc}") from exc

    try:
        dataset = GoldenDataset.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError
        raise GoldenDatasetError(f"Invalid golden dataset schema in {path}: {exc}") from exc

    csv_path = FIXTURES_DIR / dataset.csv_filename
    if not csv_path.is_file():
        raise GoldenDatasetError(f"CSV referenced by {path.name} not found: {csv_path}")

    return dataset
