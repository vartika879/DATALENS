"""In-memory registry mapping dataset_id → (DataFrame, schema).

The DataFrame is never placed in agent state. Nodes look up the actual
data through this registry using the `dataset_id` field in state.

Phase 4 will replace this with database-backed storage. The interface
(`register_dataset`, `get_dataset`) is designed to survive that swap.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pandas as pd

from datalens.schema import DatasetSchema


@dataclass
class RegisteredDataset:
    """A DataFrame together with its extracted schema."""

    df: pd.DataFrame
    schema: DatasetSchema


class DatasetNotFoundError(Exception):
    """Raised when a dataset_id is not present in the registry."""


_REGISTRY: dict[str, RegisteredDataset] = {}


def register_dataset(df: pd.DataFrame, schema: DatasetSchema) -> str:
    """Store a DataFrame and return a fresh dataset_id."""
    dataset_id = uuid4().hex
    _REGISTRY[dataset_id] = RegisteredDataset(df=df, schema=schema)
    return dataset_id


def get_dataset(dataset_id: str) -> RegisteredDataset:
    """Look up a dataset by id.

    Raises:
        DatasetNotFoundError: If the id is unknown.
    """
    if dataset_id not in _REGISTRY:
        raise DatasetNotFoundError(f"Unknown dataset_id: {dataset_id}")
    return _REGISTRY[dataset_id]


def clear_registry() -> None:
    """Remove all datasets. Intended for tests."""
    _REGISTRY.clear()
