"""Session creation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from datalens.api.auth import require_api_key
from datalens.api.rate_limit import get_rate_limiter
from datalens.api.schemas import SessionCreateRequest, SessionCreateResponse
from datalens.db.models import Dataset
from datalens.db.models import Session as DSSession
from datalens.db.session import get_db_session

router = APIRouter(tags=["session"])


@router.post("/session", response_model=SessionCreateResponse)
def create_session(
    request: SessionCreateRequest,
    db: Session = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
) -> SessionCreateResponse:
    """Create a conversation session against an existing dataset."""
    get_rate_limiter().check(api_key)
    dataset = db.get(Dataset, request.dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {request.dataset_id}",
        )

    session = DSSession(dataset_id=dataset.id)
    db.add(session)
    db.commit()
    db.refresh(session)

    return SessionCreateResponse(session_id=session.id, dataset_id=dataset.id)
