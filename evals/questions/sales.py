"""Golden questions for the sales dataset.

Each question has a `reference` callable that computes the expected answer
from the raw DataFrame.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _total_revenue(df: pd.DataFrame) -> float:
    return float((df["quantity"] * df["unit_price"]).sum())


def _highest_revenue_city(df: pd.DataFrame) -> str:
    df = df.copy()
    df["revenue"] = df["quantity"] * df["unit_price"]
    totals = df.groupby("city")["revenue"].sum()
    return str(totals.idxmax())


def _order_count(df: pd.DataFrame) -> int:
    return int(len(df))


def _avg_quantity(df: pd.DataFrame) -> float:
    return float(df["quantity"].mean())


def _unique_cities(df: pd.DataFrame) -> list[str]:
    return sorted(df["city"].dropna().unique().tolist())


QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "sales_q1",
        "question": "What is the total revenue (sum of quantity times unit_price)?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _total_revenue,
        "notes": "200+250+300+500+200+250+400+500",
    },
    {
        "id": "sales_q2",
        "question": "Which city has the highest total revenue?",
        "difficulty": "medium",
        "expected_kind": "string",
        "reference": _highest_revenue_city,
        "notes": "Mumbai=750, Delhi=950, Bangalore=900",
    },
    {
        "id": "sales_q3",
        "question": "How many orders are in the dataset?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _order_count,
    },
    {
        "id": "sales_q4",
        "question": "What is the average quantity per order?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _avg_quantity,
        "rel_tolerance": 0.001,
        "notes": "15 / 8",
    },
    {
        "id": "sales_q5",
        "question": "List all unique cities in the dataset.",
        "difficulty": "easy",
        "expected_kind": "list",
        "reference": _unique_cities,
    },
]
