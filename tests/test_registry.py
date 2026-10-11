"""Tests for the in-memory dataset registry."""

from __future__ import annotations

import pandas as pd
import pytest

from datalens.registry import (
    DatasetNotFoundError,
    clear_registry,
    get_dataset,
    register_dataset,
)
from datalens.schema import ColumnInfo, DatasetSchema


@pytest.fixture(autouse=True)
def _clean() -> None:
    clear_registry()


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3]})


@pytest.fixture
def schema() -> DatasetSchema:
    return DatasetSchema(
        file_name="t.csv",
        row_count=3,
        column_count=1,
        columns=[ColumnInfo(name="a", dtype="int64", null_count=0, null_percentage=0.0)],
        sample_rows=[{"a": 1}],
    )


def test_register_returns_unique_ids(df, schema) -> None:
    id1 = register_dataset(df, schema)
    id2 = register_dataset(df, schema)
    assert id1 != id2


def test_get_returns_registered_data(df, schema) -> None:
    did = register_dataset(df, schema)
    retrieved = get_dataset(did)
    assert retrieved.schema.file_name == "t.csv"
    assert retrieved.df.equals(df)


def test_get_unknown_id_raises() -> None:
    with pytest.raises(DatasetNotFoundError, match="Unknown dataset_id"):
        get_dataset("does-not-exist")


def test_clear_registry_removes_all(df, schema) -> None:
    did = register_dataset(df, schema)
    clear_registry()
    with pytest.raises(DatasetNotFoundError):
        get_dataset(did)
