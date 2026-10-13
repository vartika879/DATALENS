"""SQLModel tables for DataLens.

Naming notes:
- ``Session`` is a DataLens session (a conversation), not an HTTP session
  and not a SQLAlchemy session. Namespaces keep the two apart at call sites.

Design notes:
- Primary keys are UUID strings, not integers, so rows are portable across
  environments and can be generated client-side if needed.
- ``result_data_json`` is stored as JSON text. SQLite has no native JSON
  column type.
- Timestamps are UTC. ``datetime.now(UTC)`` is the source of truth.
- Relationship annotations use ``Optional["X"]``, not ``X | None``. The
  pipe syntax does not resolve through SQLAlchemy's class registry for
  forward references.
"""

from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, Relationship, SQLModel


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return uuid4().hex


# --- Dataset ---------------------------------------------------------------


class Dataset(SQLModel, table=True):
    """An uploaded CSV file and its extracted schema."""

    __tablename__ = "datasets"

    id: str = Field(default_factory=_uuid, primary_key=True)
    file_name: str
    csv_path: str
    row_count: int
    column_count: int
    uploaded_at: datetime = Field(default_factory=_now)

    sessions: list["Session"] = Relationship(
        back_populates="dataset",
        cascade_delete=True,
    )


# --- Session ---------------------------------------------------------------


class Session(SQLModel, table=True):
    """A conversation session against one dataset."""

    __tablename__ = "sessions"

    id: str = Field(default_factory=_uuid, primary_key=True)
    dataset_id: str = Field(foreign_key="datasets.id", index=True)
    created_at: datetime = Field(default_factory=_now)

    dataset: Optional["Dataset"] = Relationship(back_populates="sessions")
    runs: list["Run"] = Relationship(
        back_populates="session",
        cascade_delete=True,
    )


# --- Run -------------------------------------------------------------------


class Run(SQLModel, table=True):
    """One question → code → result turn."""

    __tablename__ = "runs"

    id: str = Field(default_factory=_uuid, primary_key=True)
    session_id: str = Field(foreign_key="sessions.id", index=True)
    question: str
    resolved_question: str | None = None
    code: str | None = None
    explanation: str | None = None
    status: str  # "success" | "error" | "timeout" | "refused" | "clarified"
    result_repr: str | None = None
    result_data_json: str | None = None  # JSON-encoded result payload
    error_message: str | None = None
    retry_count: int = 0
    latency_ms: float = 0.0
    tokens_used: int | None = None
    cost_usd: float | None = None
    created_at: datetime = Field(default_factory=_now)

    session: Optional["Session"] = Relationship(back_populates="runs")
    feedback: Optional["Feedback"] = Relationship(
        back_populates="run",
        cascade_delete=True,
    )


# --- Feedback --------------------------------------------------------------


class Feedback(SQLModel, table=True):
    """User rating on a run (thumbs up/down)."""

    __tablename__ = "feedback"

    id: str = Field(default_factory=_uuid, primary_key=True)
    run_id: str = Field(foreign_key="runs.id", index=True, unique=True)
    rating: str  # "up" | "down"
    comment: str | None = None
    created_at: datetime = Field(default_factory=_now)

    run: Optional["Run"] = Relationship(back_populates="feedback")
