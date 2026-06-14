from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, UploadFile

from src.core.models import IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_profiles(file: UploadFile) -> IngestResponse:
    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files are supported")

    try:
        content = await file.read()
        data = json.loads(content)
        if isinstance(data, dict):
            data = [data]
        return IngestResponse(
            total_profiles=len(data),
            successful=len(data),
            failed=0,
            errors=[],
        )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
