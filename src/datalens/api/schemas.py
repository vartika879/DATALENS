"""Request and response models for the DataLens API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    dataset_id: str
    file_name: str
    row_count: int
    column_count: int


class SessionCreateRequest(BaseModel):
    dataset_id: str


class SessionCreateResponse(BaseModel):
    session_id: str
    dataset_id: str


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    run_id: str
    status: str
    answer: str
    code: str | None = None
    explanation: str | None = None
    latency_ms: float
    retry_count: int = 0
