"""In-memory session registry for multi-turn conversations.

Each session holds a small history of turns (question + final answer).
The planner uses this history to resolve follow-up questions like
"ab monthly dikhao" into self-contained questions.

Phase 4 will replace this with database-backed storage. The interface
survives that swap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

MAX_HISTORY_TURNS = 5


@dataclass
class Turn:
    """One completed question-answer pair."""

    question: str
    final_answer: str


@dataclass
class Session:
    """A conversation session."""

    turns: list[Turn] = field(default_factory=list)


class SessionNotFoundError(Exception):
    """Raised when a session_id is not present in the registry."""


_SESSIONS: dict[str, Session] = {}


def create_session() -> str:
    """Create a new session and return its id."""
    session_id = uuid4().hex
    _SESSIONS[session_id] = Session()
    return session_id


def get_session(session_id: str) -> Session:
    """Look up a session by id."""
    if session_id not in _SESSIONS:
        raise SessionNotFoundError(f"Unknown session_id: {session_id}")
    return _SESSIONS[session_id]


def append_turn(session_id: str, question: str, final_answer: str) -> None:
    """Append a completed turn, trimming to MAX_HISTORY_TURNS."""
    session = get_session(session_id)
    session.turns.append(Turn(question=question, final_answer=final_answer))
    if len(session.turns) > MAX_HISTORY_TURNS:
        session.turns = session.turns[-MAX_HISTORY_TURNS:]


def get_history(session_id: str) -> list[Turn]:
    """Return a copy of the session's turns."""
    return list(get_session(session_id).turns)


def clear_sessions() -> None:
    """Remove all sessions. Intended for tests."""
    _SESSIONS.clear()
