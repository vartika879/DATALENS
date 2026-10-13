"""Tests for CSV schema extraction and validation."""

from pathlib import Path

import pytest

from datalens.schema import (
    MAX_FILE_SIZE_BYTES,
    DatasetSchema,
    SchemaExtractionError,
    extract_schema,
    validate_file_size,
)

FIXTURES = Path(__file__).parent / "fixtures"


# --- validate_file_size -----------------------------------------------------


def test_validate_file_size_accepts_valid_file(tmp_path: Path) -> None:
    csv = tmp_path / "ok.csv"
    csv.write_text("a,b\n1,2\n")
    validate_file_size(csv)  # should not raise


def test_validate_file_size_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SchemaExtractionError, match="does not exist"):
        validate_file_size(tmp_path / "nope.csv")


def test_validate_file_size_rejects_empty_file(tmp_path: Path) -> None:
    csv = tmp_path / "empty.csv"
    csv.write_text("")
    with pytest.raises(SchemaExtractionError, match="empty"):
        validate_file_size(csv)


def test_validate_file_size_rejects_oversized_file(tmp_path: Path) -> None:
    csv = tmp_path / "big.csv"
    csv.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))
    with pytest.raises(SchemaExtractionError, match="exceeds limit"):
        validate_file_size(csv)


# --- extract_schema: happy path ---------------------------------------------


def test_extract_schema_returns_expected_shape() -> None:
    schema = extract_schema(FIXTURES / "sample.csv")

    assert isinstance(schema, DatasetSchema)
    assert schema.file_name == "sample.csv"
    assert schema.row_count == 5
    assert schema.column_count == 4
    assert {c.name for c in schema.columns} == {"name", "age", "city", "salary"}


def test_extract_schema_captures_nulls() -> None:
    schema = extract_schema(FIXTURES / "sample.csv")
    by_name = {c.name: c for c in schema.columns}

    assert by_name["age"].null_count == 1
    assert by_name["salary"].null_count == 1
    assert by_name["name"].null_count == 0
    assert by_name["age"].null_percentage == pytest.approx(20.0)


def test_extract_schema_sample_rows_are_json_safe() -> None:
    schema = extract_schema(FIXTURES / "sample.csv")

    assert len(schema.sample_rows) == 5
    # Every value must be a JSON primitive or None
    for row in schema.sample_rows:
        for value in row.values():
            assert value is None or isinstance(value, (str, int, float, bool))


def test_extract_schema_nulls_become_python_none() -> None:
    schema = extract_schema(FIXTURES / "sample.csv")
    bob = next(r for r in schema.sample_rows if r["name"] == "Bob")
    charlie = next(r for r in schema.sample_rows if r["name"] == "Charlie")

    assert bob["salary"] is None
    assert charlie["age"] is None


# --- extract_schema: edge cases ---------------------------------------------


def test_extract_schema_rejects_empty_file(tmp_path: Path) -> None:
    csv = tmp_path / "empty.csv"
    csv.write_text("")
    with pytest.raises(SchemaExtractionError, match="empty"):
        extract_schema(csv)


def test_extract_schema_rejects_header_only_file(tmp_path: Path) -> None:
    csv = tmp_path / "header_only.csv"
    csv.write_text("a,b,c\n")
    schema = extract_schema(csv)

    assert schema.row_count == 0
    assert schema.column_count == 3
    assert schema.sample_rows == []


def test_extract_schema_handles_unicode_column_names(tmp_path: Path) -> None:
    csv = tmp_path / "unicode.csv"
    csv.write_text("नाम,उम्र\nराहुल,30\n", encoding="utf-8")
    schema = extract_schema(csv)

    assert {c.name for c in schema.columns} == {"नाम", "उम्र"}


# --- Hardening: encoding, BOM, columns --------------------------------------


def test_extract_schema_strips_utf8_bom(tmp_path: Path) -> None:
    """Excel adds a BOM; the first column must not start with \\ufeff."""
    csv = tmp_path / "bom.csv"
    csv.write_bytes("\ufeffname,age\nAlice,30\n".encode("utf-8"))
    schema = extract_schema(csv)
    assert {c.name for c in schema.columns} == {"name", "age"}


def test_extract_schema_falls_back_to_latin1(tmp_path: Path) -> None:
    """A file with latin-1 bytes should still parse."""
    csv = tmp_path / "latin.csv"
    csv.write_bytes("name\nJosé\n".encode("latin-1"))
    schema = extract_schema(csv)
    assert schema.column_count == 1


def test_extract_schema_rejects_duplicate_columns(tmp_path: Path) -> None:
    csv = tmp_path / "dup.csv"
    csv.write_text("a,a,b\n1,2,3\n")
    with pytest.raises(SchemaExtractionError, match="duplicate column names"):
        extract_schema(csv)


def test_extract_schema_rejects_unnamed_columns(tmp_path: Path) -> None:
    csv = tmp_path / "unnamed.csv"
    csv.write_text(",b,c\n1,2,3\n")
    with pytest.raises(SchemaExtractionError, match="unnamed columns"):
        extract_schema(csv)


def test_extract_schema_rejects_blank_column_names(tmp_path: Path) -> None:
    csv = tmp_path / "blank.csv"
    csv.write_text("a,   ,c\n1,2,3\n")
    with pytest.raises(SchemaExtractionError, match="blank column names"):
        extract_schema(csv)


def test_extract_schema_rejects_too_many_columns(tmp_path: Path) -> None:
    csv = tmp_path / "wide.csv"
    header = ",".join(f"c{i}" for i in range(600))
    row = ",".join("1" for _ in range(600))
    csv.write_text(f"{header}\n{row}\n")
    with pytest.raises(SchemaExtractionError, match="exceeds limit"):
        extract_schema(csv)
