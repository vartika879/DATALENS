"""Tests for the in-memory session registry."""

from __future__ import annotations

import pytest

from datalens.session import (
    MAX_HISTORY_TURNS,
    SessionNotFoundError,
    append_turn,
    clear_sessions,
    create_session,
    get_history,
    get_session,
)


@pytest.fixture(autouse=True)
def _clean() -> None:
    clear_sessions()


def test_create_session_returns_unique_id() -> None:
    assert create_session() != create_session()


def test_get_unknown_session_raises() -> None:
    with pytest.raises(SessionNotFoundError):
        get_session("nope")


def test_append_and_read_turn() -> None:
    sid = create_session()
    append_turn(sid, "q1", "a1")
    turns = get_history(sid)
    assert len(turns) == 1
    assert turns[0].question == "q1"
    assert turns[0].final_answer == "a1"


def test_history_is_trimmed_to_max() -> None:
    sid = create_session()
    for i in range(MAX_HISTORY_TURNS + 3):
        append_turn(sid, f"q{i}", f"a{i}")
    turns = get_history(sid)
    assert len(turns) == MAX_HISTORY_TURNS
    # Most recent should be kept
    assert turns[-1].question == f"q{MAX_HISTORY_TURNS + 2}"
