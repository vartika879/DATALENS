"""Golden questions for the Indian census dataset.

Tests state-wise aggregation, filtering, and adversarial handling.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _total_population(df: pd.DataFrame) -> int:
    return int(df["population"].sum())


def _state_count(df: pd.DataFrame) -> int:
    return int(len(df))


def _highest_population_state(df: pd.DataFrame) -> str:
    return str(df.loc[df["population"].idxmax(), "state"])


def _highest_literacy_state(df: pd.DataFrame) -> str:
    return str(df.loc[df["literacy_rate"].idxmax(), "state"])


def _average_literacy(df: pd.DataFrame) -> float:
    return float(df["literacy_rate"].mean())


def _average_sex_ratio(df: pd.DataFrame) -> float:
    return float(df["sex_ratio"].mean())


def _south_population(df: pd.DataFrame) -> int:
    return int(df[df["region"] == "South"]["population"].sum())


def _north_state_count(df: pd.DataFrame) -> int:
    return int((df["region"] == "North").sum())


def _highest_urban_percent_state(df: pd.DataFrame) -> str:
    return str(df.loc[df["urban_percent"].idxmax(), "state"])


def _states_with_literacy_above_80(df: pd.DataFrame) -> list[str]:
    return sorted(df[df["literacy_rate"] > 80]["state"].tolist())


def _unique_regions(df: pd.DataFrame) -> list[str]:
    return sorted(df["region"].dropna().unique().tolist())


def _lowest_sex_ratio_state(df: pd.DataFrame) -> str:
    return str(df.loc[df["sex_ratio"].idxmin(), "state"])


def _population_density_rank_1(df: pd.DataFrame) -> str:
    return str(df.loc[df["population"].idxmax(), "state"])


def _average_urban_percent_south(df: pd.DataFrame) -> float:
    south = df[df["region"] == "South"]
    return float(south["urban_percent"].mean())


def _profit_margin(df: pd.DataFrame) -> str:
    """Adversarial: no profit column exists in this dataset."""
    return "no profit column available"


QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "census_q1",
        "question": "What is the total population across all states?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _total_population,
    },
    {
        "id": "census_q2",
        "question": "How many states are in the dataset?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _state_count,
    },
    {
        "id": "census_q3",
        "question": "Which state has the highest population?",
        "difficulty": "easy",
        "expected_kind": "string",
        "reference": _highest_population_state,
    },
    {
        "id": "census_q4",
        "question": "Which state has the highest literacy rate?",
        "difficulty": "medium",
        "expected_kind": "string",
        "reference": _highest_literacy_state,
    },
    {
        "id": "census_q5",
        "question": "What is the average literacy rate across all states?",
        "difficulty": "medium",
        "expected_kind": "scalar",
        "reference": _average_literacy,
        "rel_tolerance": 0.005,
    },
    {
        "id": "census_q6",
        "question": "What is the average sex ratio across all states?",
        "difficulty": "medium",
        "expected_kind": "scalar",
        "reference": _average_sex_ratio,
        "rel_tolerance": 0.005,
    },
    {
        "id": "census_q7",
        "question": "What is the total population of states in the South region?",
        "difficulty": "hard",
        "expected_kind": "scalar",
        "reference": _south_population,
        "notes": "Filter by region column, then sum population.",
    },
    {
        "id": "census_q8",
        "question": "How many states are in the North region?",
        "difficulty": "medium",
        "expected_kind": "scalar",
        "reference": _north_state_count,
    },
    {
        "id": "census_q9",
        "question": "Which state has the highest urban percentage?",
        "difficulty": "medium",
        "expected_kind": "string",
        "reference": _highest_urban_percent_state,
    },
    {
        "id": "census_q10",
        "question": "List all states with a literacy rate above 80%.",
        "difficulty": "hard",
        "expected_kind": "list",
        "reference": _states_with_literacy_above_80,
        "notes": "Filter literacy_rate > 80, return state names.",
    },
    {
        "id": "census_q11",
        "question": "List all unique regions in the dataset.",
        "difficulty": "easy",
        "expected_kind": "list",
        "reference": _unique_regions,
    },
    {
        "id": "census_q12",
        "question": "Which state has the lowest sex ratio?",
        "difficulty": "medium",
        "expected_kind": "string",
        "reference": _lowest_sex_ratio_state,
    },
    {
        "id": "census_q13",
        "question": "Which state has the largest population?",
        "difficulty": "easy",
        "expected_kind": "string",
        "reference": _population_density_rank_1,
    },
    {
        "id": "census_q14",
        "question": "What is the average urban percentage across South region states?",
        "difficulty": "hard",
        "expected_kind": "scalar",
        "reference": _average_urban_percent_south,
        "rel_tolerance": 0.005,
    },
    {
        "id": "census_q15",
        "question": "What is the profit margin per state?",
        "difficulty": "adversarial",
        "expected_kind": "refusal",
        "reference": _profit_margin,
        "notes": "Adversarial: dataset has no profit column; agent should refuse.",
    },
]
