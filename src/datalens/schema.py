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
MAX_COLUMNS = 500
MAX_ROWS = 1_000_000

# --- Errors -----------------------------------------------------------------


class SchemaExtractionError(Exception):
    """Raised when a CSV cannot be validated, read, or parsed."""


class UnsupportedEncodingError(SchemaExtractionError):
    """Raised when the CSV cannot be decoded with any supported encoding."""


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


SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "latin-1")


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Read a CSV trying multiple encodings before giving up."""
    last_error: Exception | None = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.ParserError as exc:
            raise SchemaExtractionError(f"CSV parse error: {exc}") from exc
        except pd.errors.EmptyDataError as exc:
            raise SchemaExtractionError("CSV has no columns") from exc

    raise UnsupportedEncodingError(
        f"Could not decode CSV with any of {SUPPORTED_ENCODINGS}. Last error: {last_error}"
    ) from last_error


def _check_raw_header(path: Path) -> None:
    """Check the raw header line for duplicate column names.

    pandas silently renames duplicates to ``a``, ``a.1``, ``b``. We detect
    duplicates from the raw header line instead, since by the time
    ``pd.read_csv`` returns, the duplicates are gone.
    """
    for encoding in SUPPORTED_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                header_line = f.readline().rstrip("\r\n")
            break
        except UnicodeDecodeError:
            continue
    else:
        return  # decoding failed; _read_csv_with_fallback will raise

    if not header_line:
        return

    # Naive split — quoted commas in headers are rare and out of scope.
    names = [n.strip() for n in header_line.split(",")]
    seen: dict[str, int] = {}
    for name in names:
        if not name:
            continue
        seen[name] = seen.get(name, 0) + 1

    duplicates = [n for n, c in seen.items() if c > 1]
    if duplicates:
        raise SchemaExtractionError(f"CSV has duplicate column names: {duplicates[:5]}.")


def _validate_schema_limits(df: pd.DataFrame) -> None:
    """Reject DataFrames that exceed configured limits."""
    row_count = len(df)
    column_count = len(df.columns)

    if row_count > MAX_ROWS:
        raise SchemaExtractionError(f"CSV has {row_count} rows, exceeds limit of {MAX_ROWS}")
    if column_count > MAX_COLUMNS:
        raise SchemaExtractionError(
            f"CSV has {column_count} columns, exceeds limit of {MAX_COLUMNS}"
        )


def _validate_column_names(df: pd.DataFrame) -> None:
    """Reject DataFrames with duplicate, unnamed, or blank column names."""
    columns = [str(c) for c in df.columns]

    unnamed = [c for c in columns if c.startswith("Unnamed:")]
    if unnamed:
        raise SchemaExtractionError(
            f"CSV has unnamed columns: {unnamed[:5]}. "
            "Add a header row or remove the empty leading column."
        )

    blank = [c for c in columns if not c.strip()]
    if blank:
        raise SchemaExtractionError(f"CSV has blank column names ({len(blank)} of {len(columns)}).")

    seen: dict[str, int] = {}
    for name in columns:
        seen[name] = seen.get(name, 0) + 1
    duplicates = [name for name, count in seen.items() if count > 1]
    if duplicates:
        raise SchemaExtractionError(f"CSV has duplicate column names: {duplicates[:5]}.")


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
    _check_raw_header(path)

    df = _read_csv_with_fallback(path)
    _validate_schema_limits(df)
    _validate_column_names(df)

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
