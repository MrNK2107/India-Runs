from __future__ import annotations

from fastapi import APIRouter

from src.core.models import HealthResponse

router = APIRouter()

_index_size: int = 0


def init_health(index_size: int = 0) -> None:
    global _index_size
    _index_size = index_size


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        index_size=_index_size,
        models_loaded={
            "embedding": False,
            "cross_encoder": False,
        },
    )
