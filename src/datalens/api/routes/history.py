"""History endpoint — returns all runs for a session."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from datalens.api.auth import require_api_key
from datalens.api.rate_limit import get_rate_limiter
from datalens.api.schemas import HistoryResponse, RunSummary
from datalens.db.models import Run
from datalens.db.models import Session as DSSession
from datalens.db.session import get_db_session

router = APIRouter(tags=["history"])


@router.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(
    session_id: str,
    db: Session = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
) -> HistoryResponse:
    """Return all runs for a session, ordered by created_at ascending."""
    get_rate_limiter().check(api_key)

    session = db.get(DSSession, session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    runs = db.exec(
        select(Run).where(Run.session_id == session_id).order_by(Run.created_at.asc())
    ).all()

    return HistoryResponse(
        session_id=session_id,
        runs=[
            RunSummary(
                id=r.id,
                question=r.question,
                status=r.status,
                answer=r.result_repr,
                code=r.code,
                latency_ms=r.latency_ms,
                retry_count=r.retry_count,
                created_at=r.created_at,
            )
            for r in runs
        ],
    )
