"""Golden questions for the e-commerce dataset.

Emphasizes date parsing and time-series aggregation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _with_revenue(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["revenue"] = df["quantity"] * df["unit_price"]
    return df


def _total_revenue(df: pd.DataFrame) -> float:
    return float(_with_revenue(df)["revenue"].sum())


def _order_count(df: pd.DataFrame) -> int:
    return int(len(df))


def _unique_customers(df: pd.DataFrame) -> int:
    return int(df["customer"].nunique())


def _unique_products(df: pd.DataFrame) -> list[str]:
    return sorted(df["product"].dropna().unique().tolist())


def _top_product_by_revenue(df: pd.DataFrame) -> str:
    df = _with_revenue(df)
    totals = df.groupby("product")["revenue"].sum()
    return str(totals.idxmax())


def _top_customer_by_revenue(df: pd.DataFrame) -> str:
    df = _with_revenue(df)
    totals = df.groupby("customer")["revenue"].sum()
    return str(totals.idxmax())


def _avg_order_value(df: pd.DataFrame) -> float:
    df = _with_revenue(df)
    return float(df["revenue"].mean())


def _highest_single_order_revenue(df: pd.DataFrame) -> float:
    df = _with_revenue(df)
    return float(df["revenue"].max())


def _orders_in_q1(df: pd.DataFrame) -> int:
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    return int((df["order_date"].dt.quarter == 1).sum())


def _highest_revenue_month(df: pd.DataFrame) -> int:
    df = _with_revenue(df)
    df["order_date"] = pd.to_datetime(df["order_date"])
    monthly = df.groupby(df["order_date"].dt.month)["revenue"].sum()
    return int(monthly.idxmax())


def _revenue_in_march(df: pd.DataFrame) -> float:
    df = _with_revenue(df)
    df["order_date"] = pd.to_datetime(df["order_date"])
    march = df[df["order_date"].dt.month == 3]
    return float(march["revenue"].sum())


def _unique_months(df: pd.DataFrame) -> int:
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    return int(df["order_date"].dt.month.nunique())


def _largest_order_quantity(df: pd.DataFrame) -> int:
    return int(df["quantity"].max())


def _top_city_by_orders(df: pd.DataFrame) -> str:
    return str(df["city"].mode().iloc[0])


QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "ecom_q1",
        "question": "What is the total revenue (sum of quantity times unit_price)?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _total_revenue,
    },
    {
        "id": "ecom_q2",
        "question": "How many orders are in the dataset?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _order_count,
    },
    {
        "id": "ecom_q3",
        "question": "How many unique customers placed orders?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _unique_customers,
    },
    {
        "id": "ecom_q4",
        "question": "List all unique products sold.",
        "difficulty": "easy",
        "expected_kind": "list",
        "reference": _unique_products,
    },
    {
        "id": "ecom_q5",
        "question": "Which product generated the most revenue?",
        "difficulty": "medium",
        "expected_kind": "string",
        "reference": _top_product_by_revenue,
    },
    {
        "id": "ecom_q6",
        "question": "Which customer generated the most revenue?",
        "difficulty": "medium",
        "expected_kind": "string",
        "reference": _top_customer_by_revenue,
    },
    {
        "id": "ecom_q7",
        "question": "What is the average order value?",
        "difficulty": "medium",
        "expected_kind": "scalar",
        "reference": _avg_order_value,
        "rel_tolerance": 0.01,
    },
    {
        "id": "ecom_q8",
        "question": "What is the highest revenue from a single order?",
        "difficulty": "medium",
        "expected_kind": "scalar",
        "reference": _highest_single_order_revenue,
    },
    {
        "id": "ecom_q9",
        "question": "How many orders were placed in Q1 (Jan-Mar)?",
        "difficulty": "hard",
        "expected_kind": "scalar",
        "reference": _orders_in_q1,
        "notes": "Requires parsing order_date.",
    },
    {
        "id": "ecom_q10",
        "question": "Which month had the highest revenue?",
        "difficulty": "hard",
        "expected_kind": "scalar",
        "reference": _highest_revenue_month,
        "notes": "Return the month number (1-12).",
    },
    {
        "id": "ecom_q11",
        "question": "What was the total revenue in March 2024?",
        "difficulty": "hard",
        "expected_kind": "scalar",
        "reference": _revenue_in_march,
        "rel_tolerance": 0.01,
    },
    {
        "id": "ecom_q12",
        "question": "How many distinct months have orders?",
        "difficulty": "medium",
        "expected_kind": "scalar",
        "reference": _unique_months,
    },
    {
        "id": "ecom_q13",
        "question": "What is the largest quantity in any single order?",
        "difficulty": "easy",
        "expected_kind": "scalar",
        "reference": _largest_order_quantity,
    },
    {
        "id": "ecom_q14",
        "question": "Which city placed the most orders?",
        "difficulty": "medium",
        "expected_kind": "string",
        "reference": _top_city_by_orders,
    },
    {
        "id": "ecom_q15",
        "question": "What is the total profit from all orders?",
        "difficulty": "adversarial",
        "expected_kind": "refusal",
        "reference": lambda df: "no profit column available",
        "notes": "Adversarial: dataset has no profit column; agent should refuse.",
    },
]
