from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.routes.health import router as health_router
from src.api.routes.ingest import router as ingest_router
from src.api.routes.profiles import router as profiles_router
from src.api.routes.search import router as search_router
from src.core.config import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting India Runs — Intelligent Candidate Discovery")
    indexes_dir = DATA_DIR / "indexes"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="India Runs — Intelligent Candidate Discovery",
    description="Hybrid semantic search with agentic AI for candidate matching",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.include_router(search_router, prefix="/api/v1")
app.include_router(profiles_router, prefix="/api/v1")
app.include_router(ingest_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")
