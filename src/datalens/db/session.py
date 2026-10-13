"""FastAPI dependency that provides a database session per request."""

from __future__ import annotations

from collections.abc import Iterator

from sqlmodel import Session

from datalens.db.engine import get_engine


def get_db_session() -> Iterator[Session]:
    """Yield a SQLModel session and close it when the request ends."""
    with Session(get_engine()) as session:
        yield session
