"""Request and response models for the DataLens API."""

from __future__ import annotations

from datetime import datetime

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


class RunSummary(BaseModel):
    """One row in the history response."""

    id: str
    question: str
    status: str
    answer: str | None = None
    code: str | None = None
    latency_ms: float
    retry_count: int
    created_at: datetime


class HistoryResponse(BaseModel):
    session_id: str
    runs: list[RunSummary]


class FeedbackRequest(BaseModel):
    run_id: str
    rating: str = Field(..., pattern="^(up|down)$")
    comment: str | None = None


class FeedbackResponse(BaseModel):
    id: str
    run_id: str
    rating: str
    created_at: datetime
