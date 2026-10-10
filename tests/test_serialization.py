"""Tests for JSON-safety helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from datalens.serialization import sanitize_for_json


def test_none_stays_none() -> None:
    assert sanitize_for_json(None) is None


def test_nan_becomes_none() -> None:
    assert sanitize_for_json(float("nan")) is None
    assert sanitize_for_json(math.nan) is None


def test_numpy_int_becomes_python_int() -> None:
    value = sanitize_for_json(np.int64(42))
    assert value == 42
    assert type(value) is int


def test_numpy_float_becomes_python_float() -> None:
    value = sanitize_for_json(np.float64(3.14))
    assert value == pytest.approx(3.14)
    assert type(value) is float


def test_numpy_bool_becomes_python_bool() -> None:
    value = sanitize_for_json(np.bool_(True))
    assert value is True


def test_pandas_timestamp_becomes_string() -> None:
    value = sanitize_for_json(pd.Timestamp("2024-01-15"))
    assert isinstance(value, str)
    assert "2024-01-15" in value


def test_plain_scalars_are_unchanged() -> None:
    assert sanitize_for_json("hello") == "hello"
    assert sanitize_for_json(42) == 42
    assert sanitize_for_json(3.14) == pytest.approx(3.14)
    assert sanitize_for_json(True) is True
