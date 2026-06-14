from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.agents.orchestrator import Orchestrator
from src.core.models import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_orchestrator: Orchestrator | None = None


def init_orchestrator(orchestrator: Orchestrator) -> None:
    global _orchestrator
    _orchestrator = orchestrator


@router.post("/search", response_model=SearchResponse)
async def search_candidates(request: SearchRequest) -> SearchResponse:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    try:
        response = await _orchestrator.run(request.query)
        return response
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
