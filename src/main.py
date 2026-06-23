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
from src.core.profile_store import ProfileStore

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
    # Force tokenizer initialization with a dummy embed call
    _ = embedder.embed("warmup")
    set_model_loaded("embedding", True)
    logger.info("Embedding model loaded and ready")

    vector_search = VectorSearch()
    vector_search.load(faiss_path, id_map_path)
    logger.info(f"Loaded FAISS index with {vector_search.size} vectors")

    bm25_search = BM25Search()
    bm25_search.load(bm25_path)
    logger.info(f"Loaded BM25 index with {bm25_search.size} documents")

    hybrid_search = HybridSearch(vector_search, bm25_search, embedder)
    timeout_ms = get_settings().cross_encoder_timeout_ms
    reranker = CrossEncoderReranker(timeout_ms=timeout_ms)
    # Cross-encoder loading is lazy and disabled by default (CROSS_ENCODER_ENABLED=false)
    # The model will NOT be loaded at startup to avoid HuggingFace hub loops
    set_model_loaded("cross_encoder", reranker.model is not None)
    ce_status = 'loaded' if reranker.model is not None else 'disabled / unavailable'
    logger.info(f"Cross-encoder status: {ce_status}")
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
