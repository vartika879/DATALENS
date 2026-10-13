"""CSV upload endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlmodel import Session

from datalens.api.schemas import UploadResponse
from datalens.db.models import Dataset
from datalens.db.session import get_db_session
from datalens.schema import SchemaExtractionError, extract_schema
from datalens.storage import save_uploaded_csv

router = APIRouter(tags=["upload"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


def _looks_like_text(content: bytes, sample_size: int = 4096) -> bool:
    """Heuristic: reject obviously binary content."""
    sample = content[:sample_size]
    if b"\x00" in sample:
        return False
    try:
        text = sample.decode("latin-1")
    except Exception:
        return False
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\r\n\t")
    return printable / max(len(text), 1) > 0.85


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db_session),
) -> UploadResponse:
    """Accept a CSV, validate it, extract schema, and store the dataset."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are accepted.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_UPLOAD_BYTES} bytes.",
        )
    if not _looks_like_text(content):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File does not look like text (possibly binary).",
        )
    csv_path = save_uploaded_csv(file.filename, content)

    try:
        schema = extract_schema(csv_path)
    except SchemaExtractionError as exc:
        csv_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid CSV: {exc}",
        ) from exc

    dataset = Dataset(
        file_name=file.filename or schema.file_name,
        csv_path=str(csv_path),
        row_count=schema.row_count,
        column_count=schema.column_count,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return UploadResponse(
        dataset_id=dataset.id,
        file_name=dataset.file_name,
        row_count=dataset.row_count,
        column_count=dataset.column_count,
    )
