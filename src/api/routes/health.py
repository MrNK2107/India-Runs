from __future__ import annotations

from fastapi import APIRouter

from src.core.models import HealthResponse

router = APIRouter()

_index_size: int = 0
_models_loaded: dict[str, bool] = {"embedding": False, "cross_encoder": False}


def init_health(index_size: int = 0) -> None:
    global _index_size, _models_loaded
    _index_size = index_size
    _models_loaded = {"embedding": False, "cross_encoder": False}


def set_index_size(size: int) -> None:
    global _index_size
    _index_size = size


def set_model_loaded(name: str, loaded: bool = True) -> None:
    _models_loaded[name] = loaded


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy" if _index_size > 0 else "degraded",
        version="0.1.0",
        index_size=_index_size,
        models_loaded=dict(_models_loaded),
    )
