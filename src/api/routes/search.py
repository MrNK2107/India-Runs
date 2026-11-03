from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from src.agents.orchestrator import Orchestrator
from src.core.models import ExplainRequest, Explanation, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_orchestrator: Orchestrator | None = None


def init_orchestrator(orchestrator: Orchestrator | None) -> None:
    global _orchestrator
    _orchestrator = orchestrator


@router.post("/search", response_model=SearchResponse)
async def search_candidates(request: SearchRequest) -> SearchResponse:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    include_current = request.trajectory_mode == "include_current"
    try:
        response = await _orchestrator.run(
            raw_query=request.query,
            slider_weights=request.slider_weights,
            use_turbo=request.use_turbo,
            top_k=request.max_results,
            filters=request.filters,
            include_current_tenure=include_current,
            credit_work_history=request.skill_evidence == "work_history",
        )
        executor = getattr(_orchestrator, "executor", None)
        if request.explain and executor is not None:
            from src.explain.service import attach_explanations

            try:
                await run_in_threadpool(
                    attach_explanations,
                    response, executor, request.query, request.filters,
                    request.slider_weights, include_current,
                    request.skill_evidence == "work_history",
                )
            except Exception:
                # Explanations are additive: never fail a search because of them.
                logger.exception("Explanations failed; returning results without them")
        return response
    except Exception as e:
        # Log the full error server-side; don't leak internals (paths, model
        # errors, provider messages) to API clients.
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail="Search failed") from e


@router.post("/explain", response_model=Explanation)
async def explain(request: ExplainRequest) -> Explanation:
    """Why (or why not) this candidate for this query: score contributions,
    requirement checks with quoted evidence, strengths and risks."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    from src.explain.service import explain_profile

    try:
        result = await run_in_threadpool(
            explain_profile,
            _orchestrator.executor, request.query, request.profile_id,
            request.slider_weights, request.trajectory_mode == "include_current",
            request.skill_evidence == "work_history",
        )
    except Exception as e:
        logger.exception("Explain failed")
        raise HTTPException(status_code=500, detail="Explain failed") from e
    if result is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return result
