"""Ask endpoint — runs the pipeline and records a Run row."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from datalens.api.auth import require_api_key
from datalens.api.rate_limit import get_rate_limiter
from datalens.api.schemas import AskRequest, AskResponse
from datalens.db.models import Dataset, Run
from datalens.db.models import Session as DSSession
from datalens.db.session import get_db_session
from datalens.pipeline import run_pipeline
from datalens.schema import extract_schema

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    db: Session = Depends(get_db_session),
    api_key: str = Depends(require_api_key),
) -> AskResponse:
    """Run one single-turn question against a session's dataset."""
    get_rate_limiter().check(api_key)
    session = db.get(DSSession, request.session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {request.session_id}",
        )

    dataset = db.get(Dataset, session.dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Session references a missing dataset.",
        )

    csv_path = Path(dataset.csv_path)
    if not csv_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dataset file is no longer available.",
        )

    try:
        df = pd.read_csv(csv_path)
        schema = extract_schema(csv_path)
    except Exception as exc:  # noqa: BLE001 — surface as 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load dataset: {exc}",
        ) from exc

    result = run_pipeline(schema, request.question, df)

    # Persist a Run row.
    run = Run(
        session_id=session.id,
        question=request.question,
        resolved_question=result.generated is not None and request.question or None,
        code=result.generated.code if result.generated else None,
        explanation=result.generated.explanation if result.generated else None,
        status=result.status,
        result_repr=result.execution.result_repr if result.execution else None,
        result_data_json=(
            json.dumps(result.execution.result_data)
            if result.execution and result.execution.result_data is not None
            else None
        ),
        error_message=result.error_message,
        retry_count=0,
        latency_ms=result.total_duration_ms,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Build the user-facing answer.
    if result.status == "success" and result.execution is not None:
        answer = result.execution.result_repr or "(no result)"
    else:
        answer = result.error_message or "The request could not be completed."

    return AskResponse(
        run_id=run.id,
        status=run.status,
        answer=answer,
        code=run.code,
        explanation=run.explanation,
        latency_ms=run.latency_ms,
        retry_count=run.retry_count,
    )
