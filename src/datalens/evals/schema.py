"""Pydantic models for the golden evaluation dataset.

A golden dataset is a versioned, human-verified set of (question, expected
answer) pairs. It is the source of truth for accuracy measurements.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Difficulty(StrEnum):
    """Difficulty bucket for a golden question."""

    EASY = "easy"  # single aggregation
    MEDIUM = "medium"  # groupby, filter
    HARD = "hard"  # joins, window, time series
    ADVERSARIAL = "adversarial"  # wrong column, ambiguous, unanswerable


class AnswerKind(StrEnum):
    """Shape of the expected answer."""

    SCALAR = "scalar"  # int or float
    STRING = "string"
    LIST = "list"  # order-independent after sorting
    BOOL = "bool"


class GoldenQuestion(BaseModel):
    """One row in the golden dataset."""

    id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    difficulty: Difficulty
    expected_kind: AnswerKind
    expected: Any
    rel_tolerance: float = Field(default=0.01, ge=0.0)
    abs_tolerance: float = Field(default=1e-9, ge=0.0)
    notes: str | None = None


class GoldenDataset(BaseModel):
    """A complete evaluation dataset for one CSV."""

    name: str = Field(..., min_length=1)
    csv_filename: str = Field(..., min_length=1)
    description: str = ""
    questions: list[GoldenQuestion] = Field(..., min_length=1)
