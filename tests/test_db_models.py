"""Tests for SQLModel tables and basic CRUD."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, select

from datalens.db.engine import get_engine
from datalens.db.models import Dataset, Feedback, Run
from datalens.db.models import Session as DSSession


@pytest.fixture(autouse=True)
def _in_memory_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a fresh in-memory SQLite database for each test."""
    monkeypatch.setenv("DATALENS_DB_URL", "sqlite:///:memory:")
    get_engine.cache_clear()
    # Import so metadata is populated.
    from datalens.db import models  # noqa: F401

    SQLModel.metadata.create_all(get_engine())
    yield
    get_engine.cache_clear()


def _make_dataset(session: Session) -> Dataset:
    ds = Dataset(
        file_name="sample.csv",
        csv_path="/tmp/sample.csv",
        row_count=5,
        column_count=3,
    )
    session.add(ds)
    session.commit()
    session.refresh(ds)
    return ds


def test_dataset_roundtrip() -> None:
    with Session(get_engine()) as s:
        ds = _make_dataset(s)
        assert ds.id
        assert ds.uploaded_at is not None
        found = s.exec(select(Dataset)).one()
        assert found.file_name == "sample.csv"


def test_session_links_to_dataset() -> None:
    with Session(get_engine()) as s:
        ds = _make_dataset(s)
        sess = DSSession(dataset_id=ds.id)
        s.add(sess)
        s.commit()
        s.refresh(sess)
        assert sess.id
        assert sess.dataset_id == ds.id


def test_run_linked_to_session() -> None:
    with Session(get_engine()) as s:
        ds = _make_dataset(s)
        sess = DSSession(dataset_id=ds.id)
        s.add(sess)
        s.commit()
        s.refresh(sess)

        run = Run(
            session_id=sess.id,
            question="total salary?",
            status="success",
            latency_ms=123.4,
            result_repr="57500.0",
            result_data_json='{"value": 57500.0}',
        )
        s.add(run)
        s.commit()
        s.refresh(run)

        assert run.id
        assert run.created_at is not None
        assert run.retry_count == 0


def test_feedback_links_to_run() -> None:
    with Session(get_engine()) as s:
        ds = _make_dataset(s)
        sess = DSSession(dataset_id=ds.id)
        s.add(sess)
        s.commit()
        s.refresh(sess)

        run = Run(session_id=sess.id, question="q", status="success")
        s.add(run)
        s.commit()
        s.refresh(run)

        fb = Feedback(run_id=run.id, rating="up", comment="nice")
        s.add(fb)
        s.commit()
        s.refresh(fb)

        assert fb.id
        assert fb.run_id == run.id


def test_feedback_unique_per_run() -> None:
    """Only one feedback row per run (unique constraint)."""
    with Session(get_engine()) as s:
        ds = _make_dataset(s)
        sess = DSSession(dataset_id=ds.id)
        s.add(sess)
        s.commit()
        s.refresh(sess)
        run = Run(session_id=sess.id, question="q", status="success")
        s.add(run)
        s.commit()
        s.refresh(run)

        s.add(Feedback(run_id=run.id, rating="up"))
        s.commit()

        s.add(Feedback(run_id=run.id, rating="down"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_create_db_and_tables_is_idempotent() -> None:
    from datalens.db.engine import create_db_and_tables

    create_db_and_tables()  # second call must not raise
    create_db_and_tables()
