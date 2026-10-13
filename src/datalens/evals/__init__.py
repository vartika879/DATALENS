"""Evaluation framework for DataLens.

Provides golden-dataset loading, answer comparison, and (in later tasks)
a runner that executes questions end-to-end and reports metrics.
"""

from datalens.evals.compare import answers_match
from datalens.evals.loader import load_golden_dataset
from datalens.evals.schema import (
    AnswerKind,
    Difficulty,
    GoldenDataset,
    GoldenQuestion,
)

__all__ = [
    "AnswerKind",
    "Difficulty",
    "GoldenDataset",
    "GoldenQuestion",
    "answers_match",
    "load_golden_dataset",
]
