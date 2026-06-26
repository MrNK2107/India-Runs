from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

# Force offline mode globally to prevent HuggingFace hub HEAD loops
# Must be set before any sentence-transformers imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI

from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.validation import InputValidationMiddleware
from src.api.routes.health import init_health, set_index_size, set_model_loaded
from src.api.routes.health import router as health_router
from src.api.routes.ingest import router as ingest_router
from src.api.routes.profiles import init_profiles
from src.api.routes.profiles import router as profiles_router
from src.api.routes.search import init_orchestrator
from src.api.routes.search import router as search_router
from src.core.config import DATA_DIR, get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _start_time = time.time()
    init_health(index_size=0)
    logger.info("Starting India Runs — Intelligent Candidate Discovery")
    indexes_dir = DATA_DIR / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = indexes_dir / "faiss_index.bin"
    id_map_path = indexes_dir / "faiss_id_map.json"
    bm25_path = indexes_dir / "bm25_index.pkl"

    if not faiss_path.exists():
        logger.warning(
            "No FAISS index found. Run 'python scripts/build_indexes.py --sample 50' first."
        )
        init_health(index_size=0)
        yield
        return

    from src.core.config import build_orchestrator

    orchestrator, vector_search, profiles = build_orchestrator(
        faiss_path=faiss_path,
        id_map_path=id_map_path,
        bm25_path=bm25_path,
        cross_encoder_timeout_ms=get_settings().cross_encoder_timeout_ms,
    )
    set_model_loaded("embedding", True)
    set_model_loaded("cross_encoder", True)  # already handled inside build_orchestrator

    init_orchestrator(orchestrator)
    set_index_size(vector_search.size)
    init_profiles(profiles)

    logger.info("System initialized successfully (%.1fs)", time.time() - _start_time)

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="India Runs — Intelligent Candidate Discovery",
    description="Hybrid semantic search with agentic AI for candidate matching",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(InputValidationMiddleware)
app.include_router(search_router, prefix="/api/v1")
app.include_router(profiles_router, prefix="/api/v1")
app.include_router(ingest_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, log_level="info")
