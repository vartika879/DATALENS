"""Golden questions for the Titanic dataset.

Each question has a `reference` callable that computes the expected answer
from the raw DataFrame. The generator script uses these to build the golden
JSON, and the consistency test verifies no drift.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _survived_count(df: pd.DataFrame) -> int:
    return int(df["survived"].sum())


def _survival_rate(df: pd.DataFrame) -> float:
    return float(df["survived"].mean())


def _female_count(df: pd.DataFrame) -> int:
    return int((df["sex"] == "female").sum())


def _female_survival_rate(df: pd.DataFrame) -> float:
    return float(df[df["sex"] == "female"]["survived"].mean())


def _male_survival_rate(df: pd.DataFrame) -> float:
    return float(df[df["sex"] == "male"]["survived"].mean())


def _average_age(df: pd.DataFrame) -> float:
    return float(df["age"].mean())


def _average_fare(df: pd.DataFrame) -> float:
    return float(df["fare"].mean())


def _highest_survival_pclass(df: pd.DataFrame) -> int:
    rates = df.groupby("pclass")["survived"].mean()
    return int(rates.idxmax())


def _first_class_count(df: pd.DataFrame) -> int:
    return int((df["pclass"] == 1).sum())


def _unique_embarked(df: pd.DataFrame) -> list[str]:
    return sorted(df["embarked"].dropna().unique().tolist())


def _oldest_survivor_age(df: pd.DataFrame) -> float:
    return float(df[df["survived"] == 1]["age"].max())


def _youngest_passenger_age(df: pd.DataFrame) -> float:
    return float(df["age"].min())


def _missing_age_count(df: pd.DataFrame) -> int:
    return int(df["age"].isna().sum())


def _total_fare_collected(df: pd.DataFrame) -> float:
    return float(df["fare"].sum())


def _most_common_embarkation(df: pd.DataFrame) -> str:
    return str(df["embarked"].mode().iloc[0])


QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "titanic_q1",
        "question": "How many passengers survived?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _survived_count,
    },
    {
        "id": "titanic_q2",
        "question": "What fraction of passengers survived?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _survival_rate,
        "rel_tolerance": 0.001,
    },
    {
        "id": "titanic_q3",
        "question": "How many passengers were female?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _female_count,
    },
    {
        "id": "titanic_q4",
        "question": "What was the survival rate among female passengers?",
        "difficulty": "medium",
        "expected_kind": "scalar",
        "reference": _female_survival_rate,
        "rel_tolerance": 0.001,
    },
    {
        "id": "titanic_q5",
        "question": "What was the survival rate among male passengers?",
        "difficulty": "medium",
        "expected_kind": "scalar",
        "reference": _male_survival_rate,
        "rel_tolerance": 0.001,
    },
    {
        "id": "titanic_q6",
        "question": "What was the average age of passengers?",
        "difficulty": "medium",
        "expected_kind": "scalar",
        "reference": _average_age,
        "rel_tolerance": 0.01,
    },
    {
        "id": "titanic_q7",
        "question": "What was the average fare paid?",
        "difficulty": "medium",
        "expected_kind": "scalar",
        "reference": _average_fare,
        "rel_tolerance": 0.01,
    },
    {
        "id": "titanic_q8",
        "question": "Which passenger class had the highest survival rate?",
        "difficulty": "hard",
        "expected_kind": "scalar",
        "reference": _highest_survival_pclass,
        "notes": "Return the pclass number (1, 2, or 3).",
    },
    {
        "id": "titanic_q9",
        "question": "How many passengers were in first class?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _first_class_count,
    },
    {
        "id": "titanic_q10",
        "question": "List all unique embarkation ports in alphabetical order.",
        "difficulty": "medium",
        "expected_kind": "list",
        "reference": _unique_embarked,
    },
    {
        "id": "titanic_q11",
        "question": "What was the age of the oldest passenger who survived?",
        "difficulty": "hard",
        "expected_kind": "scalar",
        "reference": _oldest_survivor_age,
    },
    {
        "id": "titanic_q12",
        "question": "What was the age of the youngest passenger?",
        "difficulty": "medium",
        "expected_kind": "scalar",
        "reference": _youngest_passenger_age,
    },
    {
        "id": "titanic_q13",
        "question": "How many passengers have a missing age value?",
        "difficulty": "medium",
        "expected_kind": "scalar",
        "reference": _missing_age_count,
    },
    {
        "id": "titanic_q14",
        "question": "What was the total fare collected across all passengers?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _total_fare_collected,
        "rel_tolerance": 0.01,
    },
    {
        "id": "titanic_q15",
        "question": "Which embarkation port was the most common?",
        "difficulty": "medium",
        "expected_kind": "string",
        "reference": _most_common_embarkation,
    },
]
