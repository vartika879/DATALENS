"""CSV schema extraction and validation.

This module turns a CSV file into a compact, JSON-safe description
that can be safely sent to an LLM. It never sends raw file contents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from datalens.serialization import sanitize_for_json

# --- Limits and constants ---------------------------------------------------

MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SAMPLE_ROW_COUNT = 5


# --- Errors -----------------------------------------------------------------


class SchemaExtractionError(Exception):
    """Raised when a CSV cannot be validated, read, or parsed."""


# --- Models -----------------------------------------------------------------


class ColumnInfo(BaseModel):
    """Metadata about a single CSV column."""

    name: str
    dtype: str
    null_count: int = Field(ge=0)
    null_percentage: float = Field(ge=0.0, le=100.0)


class DatasetSchema(BaseModel):
    """Compact, LLM-safe description of a CSV dataset.

    Contains only structure and a few sample rows — never the full data.
    """

    file_name: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[ColumnInfo]
    sample_rows: list[dict[str, Any]]


def validate_file_size(path: Path) -> None:
    """Check the file exists, is non-empty, and is within the size limit.

    Raises:
        SchemaExtractionError: If any check fails.
    """
    if not path.exists():
        raise SchemaExtractionError(f"File does not exist: {path}")
    if not path.is_file():
        raise SchemaExtractionError(f"Path is not a file: {path}")

    size = path.stat().st_size
    if size == 0:
        raise SchemaExtractionError("File is empty")
    if size > MAX_FILE_SIZE_BYTES:
        raise SchemaExtractionError(
            f"File size {size} bytes exceeds limit of "
            f"{MAX_FILE_SIZE_BYTES} bytes ({MAX_FILE_SIZE_MB} MB)"
        )


# --- Internal helpers -------------------------------------------------------


def _sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(k): sanitize_for_json(v) for k, v in row.items()}


# --- Public API -------------------------------------------------------------


def extract_schema(path: Path) -> DatasetSchema:
    """Validate, load, and summarize a CSV file.

    Args:
        path: Path to a CSV file.

    Returns:
        A DatasetSchema with column metadata and up to SAMPLE_ROW_COUNT rows.

    Raises:
        SchemaExtractionError: On validation, parsing, or encoding failures.
    """
    validate_file_size(path)

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise SchemaExtractionError("CSV has no columns") from exc
    except pd.errors.ParserError as exc:
        raise SchemaExtractionError(f"CSV parse error: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise SchemaExtractionError(f"Encoding error: {exc}") from exc

    row_count = len(df)
    denominator = max(row_count, 1)  # avoid zero-division on header-only files

    columns: list[ColumnInfo] = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        columns.append(
            ColumnInfo(
                name=str(col),
                dtype=str(df[col].dtype),
                null_count=null_count,
                null_percentage=round(100.0 * null_count / denominator, 2),
            )
        )

    raw_samples = df.head(SAMPLE_ROW_COUNT).to_dict(orient="records")
    sample_rows = [_sanitize_row(row) for row in raw_samples]

    return DatasetSchema(
        file_name=path.name,
        row_count=row_count,
        column_count=len(df.columns),
        columns=columns,
        sample_rows=sample_rows,
    )
