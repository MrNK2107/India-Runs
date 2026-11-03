from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from src.core.models import IngestResponse
from src.ingestion.service import MAX_PROFILES_PER_REQUEST, IngestService

logger = logging.getLogger(__name__)

router = APIRouter()

# Same as InputValidationMiddleware.MAX_BODY_SIZE, but enforced on the bytes actually read:
# the middleware only sees Content-Length, which chunked uploads don't send.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

_ingest_service: IngestService | None = None


def init_ingest(service: IngestService | None) -> None:
    global _ingest_service
    _ingest_service = service


def _reject_constant(name: str) -> Any:
    # json.loads accepts NaN/Infinity by default; they are not valid JSON.
    raise ValueError(f"{name} is not valid JSON")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_profiles(file: UploadFile) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(
            status_code=400,
            detail=f"Only .json files are supported (got {file.filename or 'no filename'!r})",
        )

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES} bytes)"
            )
        data: Any = json.loads(content, parse_constant=_reject_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        # Scalars (e.g. `5`, `"x"`, `null`) used to crash with TypeError -> 500.
        raise HTTPException(
            status_code=400, detail="JSON must be a profile object or a list of profiles"
        )
    if len(data) > MAX_PROFILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many profiles: {len(data)} (max {MAX_PROFILES_PER_REQUEST} per request)",
        )

    if _ingest_service is None:
        raise HTTPException(
            status_code=503,
            detail="Search index not initialized; build indexes first "
            "(python scripts/build_indexes.py --sample 50)",
        )
    try:
        # Embedding + BM25 rebuild are CPU-bound: keep them off the event loop.
        result = await run_in_threadpool(_ingest_service.ingest, data)
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail="Ingest failed") from e

    return IngestResponse(
        total_profiles=len(data),
        successful=result.successful,
        failed=len(result.errors),
        created=len(result.created),
        updated=len(result.updated),
        unchanged=len(result.unchanged),
        profile_ids=[*result.created, *result.updated, *result.unchanged],
        errors=result.errors,
    )
