from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.routes.health import init_health
from src.api.routes.health import router as health_router
from src.api.routes.ingest import router as ingest_router
from src.api.routes.profiles import init_profiles
from src.api.routes.profiles import router as profiles_router
from src.api.routes.search import init_orchestrator
from src.api.routes.search import router as search_router
from src.core.config import DATA_DIR
from src.core.profile_store import ProfileStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
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

    from src.agents.executor import ExecutorAgent
    from src.agents.orchestrator import Orchestrator
    from src.agents.planner import PlannerAgent
    from src.agents.reflector import ReflectorAgent
    from src.language.multilingual import MultilingualEmbedder
    from src.matching.scorer import CandidateScorer
    from src.search.bm25_search import BM25Search
    from src.search.hybrid import HybridSearch
    from src.search.reranker import CrossEncoderReranker
    from src.search.vector_search import VectorSearch

    embedder = MultilingualEmbedder()
    logger.info("Pre-loading embedding model...")
    _ = embedder.model
    logger.info("Embedding model loaded")

    vector_search = VectorSearch()
    vector_search.load(faiss_path, id_map_path)
    logger.info(f"Loaded FAISS index with {vector_search.size} vectors")

    bm25_search = BM25Search()
    bm25_search.load(bm25_path)
    logger.info(f"Loaded BM25 index with {bm25_search.size} documents")

    hybrid_search = HybridSearch(vector_search, bm25_search, embedder)
    reranker = CrossEncoderReranker(timeout_ms=500)
    logger.info("Pre-loading cross-encoder model...")
    _ = reranker.model
    logger.info("Cross-encoder model loaded")
    scorer = CandidateScorer()

    profiles = ProfileStore()
    sample_path = DATA_DIR / "samples" / "sample_candidates.json"
    if sample_path.exists():
        profiles.load_sample(sample_path)

    offset_index_path = DATA_DIR / "indexes" / "offset_index.json"
    if offset_index_path.exists():
        profiles.load_offset_index(offset_index_path)
        logger.info(f"ProfileStore ready: {len(profiles)} profiles available")
    else:
        logger.info("No offset index found — profiles will be indexed on first access")
    logger.info("ProfileStore initialized (lazy load)")

    planner = PlannerAgent()
    executor = ExecutorAgent(hybrid_search, reranker, scorer, profiles)
    reflector = ReflectorAgent()
    orchestrator = Orchestrator(planner, executor, reflector)

    init_orchestrator(orchestrator)
    init_health(index_size=vector_search.size)
    init_profiles(profiles)

    logger.info("System initialized successfully (%.1fs)", 0.0)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, log_level="info")
