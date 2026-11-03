from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Force offline mode globally to prevent HuggingFace hub HEAD loops
# Must be set before any sentence-transformers imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.errors import validation_exception_handler
from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.validation import InputValidationMiddleware
from src.api.routes.fairness import router as fairness_router
from src.api.routes.health import (
    init_health,
    set_engine_info,
    set_index_size,
    set_model_loaded,
)
from src.api.routes.health import router as health_router
from src.api.routes.ingest import init_ingest
from src.api.routes.ingest import router as ingest_router
from src.api.routes.profiles import init_profiles
from src.api.routes.profiles import router as profiles_router
from src.api.routes.search import init_orchestrator
from src.api.routes.search import router as search_router
from src.core.config import DATA_DIR, get_settings
from src.core.constants import INGESTED_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _after_ingest(hybrid: Any, vector_search: Any) -> None:
    hybrid._cache.clear()  # cached result lists don't include new profiles
    set_index_size(vector_search.size)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _start_time = time.time()
    init_health(index_size=0)
    logger.info("Starting Kasauti (india-runs) - evidence-backed candidate ranking")
    indexes_dir = DATA_DIR / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = indexes_dir / "faiss_index.bin"
    id_map_path = indexes_dir / "faiss_id_map.json"
    bm25_path = indexes_dir / "bm25_index.pkl"

    prebuilt = None
    sample_profiles: list[Any] = []
    if not faiss_path.exists():
        if os.environ.get("AUTO_INDEX", "true").strip().lower() in ("0", "false", "no"):
            logger.warning(
                "No FAISS index found. Run 'python scripts/build_indexes.py --sample 50' first."
            )
            init_health(index_size=0)
            yield
            return
        # Zero-setup mode: index the bundled sample in memory (see src/core/engine.py).
        from src.core.engine import DEFAULT_SAMPLE, build_sample_engine

        sample = Path(os.environ.get("SAMPLE_PROFILES_PATH", DEFAULT_SAMPLE))
        try:
            embedder, vs, bm25, sample_profiles = build_sample_engine(sample)
        except Exception:
            # e.g. transformer backend with no cached model (HF is offline here) or a bad
            # SAMPLE_PROFILES_PATH: start degraded, as without AUTO_INDEX, not crash.
            logger.exception(
                "Could not build the sample index; starting degraded. "
                "Set EMBEDDING_BACKEND=hashing or build indexes with scripts/build_indexes.py."
            )
            init_health(index_size=0)
            yield
            return
        prebuilt = (embedder, vs, bm25)
        set_engine_info(mode="sample", dataset=sample.name)
    else:
        set_engine_info(mode="index", dataset=faiss_path.name)

    from src.core.config import build_orchestrator

    orchestrator, vector_search, profiles = build_orchestrator(
        faiss_path=faiss_path,
        id_map_path=id_map_path,
        bm25_path=bm25_path,
        cross_encoder_timeout_ms=get_settings().cross_encoder_timeout_ms,
        prebuilt=prebuilt,
    )
    if sample_profiles:
        profiles.add_samples(sample_profiles)
    set_model_loaded("embedding", True)
    set_model_loaded("cross_encoder", get_settings().cross_encoder_enabled)

    init_orchestrator(orchestrator)
    init_profiles(profiles)

    from src.ingestion.service import IngestService

    hybrid = orchestrator.executor.hybrid_search
    ingest_service = IngestService(
        profile_store=profiles,
        vector_search=vector_search,
        bm25_search=hybrid.bm25_search,
        embedder=hybrid.embedder,
        store_path=Path(os.environ.get("INGESTED_PROFILES_PATH", INGESTED_PATH)),
        on_change=lambda: _after_ingest(hybrid, vector_search),
    )
    ingest_service.load()  # replay profiles ingested in earlier runs
    init_ingest(ingest_service)
    set_index_size(vector_search.size)

    logger.info("System initialized successfully (%.1fs)", time.time() - _start_time)

    yield

    logger.info("Shutting down")


WEB_DIR = Path(__file__).resolve().parent / "web" / "static"

app = FastAPI(
    title="Kasauti — evidence-backed candidate ranking",
    description=(
        "Hybrid semantic search over candidate profiles. Every result carries an "
        "explanation: score contributions, requirement checks and quoted evidence."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(InputValidationMiddleware)
app.include_router(search_router, prefix="/api/v1")
app.include_router(profiles_router, prefix="/api/v1")
app.include_router(ingest_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")
app.include_router(fairness_router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, log_level="info")
