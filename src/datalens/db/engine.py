"""Database engine and schema initialization.

The default engine points at ``data/datalens.db``. Tests can override by
calling ``get_engine.cache_clear()`` and setting ``DATALENS_DB_URL``.

Note on in-memory SQLite:
    ``sqlite:///:memory:`` normally gives each *connection* its own empty
    database. With SQLAlchemy's default pooling, ``create_all()`` runs on
    one connection and later queries hit a fresh, empty one. We use
    ``StaticPool`` for in-memory URLs so every session shares the same
    underlying connection and the schema persists across the test.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

DEFAULT_DB_PATH = Path("data") / "datalens.db"
IN_MEMORY_URL = "sqlite:///:memory:"


def _default_url() -> str:
    """Return the SQLite URL for the current environment."""
    override = os.environ.get("DATALENS_DB_URL")
    if override:
        return override
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB_PATH}"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the singleton SQLAlchemy engine."""
    url = _default_url()

    # In-memory SQLite needs StaticPool so all sessions share one connection.
    if url == IN_MEMORY_URL:
        return create_engine(
            url,
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

    # File-backed SQLite requires check_same_thread=False for FastAPI threads.
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, echo=False, connect_args=connect_args)


def create_db_and_tables() -> None:
    """Create all tables if they do not exist. Idempotent."""
    # Import models so SQLModel sees the metadata before create_all runs.
    from datalens.db import models  # noqa: F401

    SQLModel.metadata.create_all(get_engine())
