"""Feedback endpoint — capture user ratings on runs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from datalens.api.auth import require_api_key
from datalens.api.rate_limit import get_rate_limiter
from datalens.api.schemas import FeedbackRequest, FeedbackResponse
from datalens.db.models import Feedback, Run
from datalens.db.session import get_db_session

router = APIRouter(tags=["feedback"])


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
) -> FeedbackResponse:
    """Record user feedback on a run. One feedback per run (unique)."""
    get_rate_limiter().check(api_key)

    run = db.get(Run, request.run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {request.run_id}",
        )

    existing = db.exec(select(Feedback).where(Feedback.run_id == request.run_id)).first()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Feedback already submitted for this run.",
        )

    feedback = Feedback(
        run_id=request.run_id,
        rating=request.rating,
        comment=request.comment,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    return FeedbackResponse(
        id=feedback.id,
        run_id=feedback.run_id,
        rating=feedback.rating,
        created_at=feedback.created_at,
    )
