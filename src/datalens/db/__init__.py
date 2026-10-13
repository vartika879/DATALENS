"""Persistence layer for DataLens.

Uses SQLModel (Pydantic + SQLAlchemy) over SQLite. The DB is intentionally
local and file-backed for now; Phase 7 will move this to a hosted database
if the deployment needs it.
"""

from datalens.db.engine import create_db_and_tables, get_engine
from datalens.db.models import Dataset, Feedback, Run, Session

__all__ = [
    "Dataset",
    "Feedback",
    "Run",
    "Session",
    "create_db_and_tables",
    "get_engine",
]
