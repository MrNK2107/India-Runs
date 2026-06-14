"""FastAPI application entry point — placeholder for Phase 10."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Starting India Runs — Intelligent Candidate Discovery")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="India Runs — Intelligent Candidate Discovery",
    description="Hybrid semantic search with agentic AI for candidate matching",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Health check ping."""
    return {"status": "ok", "message": "India Runs is running"}
