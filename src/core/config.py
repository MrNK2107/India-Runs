from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres@localhost:5432/india_runs"  # set DATABASE_URL (with password) in .env
    redis_url: str = "redis://localhost:6379/0"
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    log_level: str = "INFO"
    max_replan_cycles: int = 1
    cross_encoder_timeout_ms: int = 0
    # CROSS_ENCODER_ENABLED=false skips loading the reranker model (saves
    # ~300 MB); rerank then falls back to the hybrid retrieval scores.
    cross_encoder_enabled: bool = True

    model_config = {"env_file": ".env", "extra": "ignore"}


def load_yaml_config(filename: str) -> dict[str, Any]:
    path = CONFIGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    # Explicit UTF-8: the default (locale) encoding is cp1252 on Windows and the
    # configs contain non-ASCII characters.
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping, got {type(data).__name__}")
    return data


@cache
def get_settings() -> Settings:
    return Settings()


@cache
def get_scoring_config() -> dict[str, Any]:
    return load_yaml_config("scoring_weights.yaml")


@cache
def get_model_config() -> dict[str, Any]:
    return load_yaml_config("models.yaml")


@cache
def get_app_config() -> dict[str, Any]:
    return load_yaml_config("settings.yaml")


def build_orchestrator(
    faiss_path: Path,
    id_map_path: Path,
    bm25_path: Path,
    cross_encoder_timeout_ms: int = 0,
    prebuilt: tuple[Any, Any, Any] | None = None,
) -> tuple[Any, Any, Any]:
    """Build the full search dependency chain.

    Shared between main.py (FastAPI lifespan) and app.py (Gradio UI)
    to avoid duplicating the ~40-line initialization block.

    ``prebuilt`` = (embedder, VectorSearch, BM25Search) built in memory (the sample/demo
    engine, see src/core/engine.py); otherwise the indexes are loaded from disk.

    Returns (Orchestrator, VectorSearch, ProfileStore).
    """
    from src.agents.executor import ExecutorAgent
    from src.agents.orchestrator import Orchestrator
    from src.agents.planner import PlannerAgent
    from src.agents.reflector import ReflectorAgent
    from src.core.profile_store import ProfileStore
    from src.language.multilingual import create_embedder
    from src.matching.scorer import CandidateScorer
    from src.search.bm25_search import BM25Search
    from src.search.hybrid import HybridSearch
    from src.search.reranker import CrossEncoderReranker
    from src.search.vector_search import VectorSearch

    if prebuilt is not None:
        embedder, vector_search, bm25_search = prebuilt
    else:
        embedder = create_embedder()
        _ = embedder.model
        _ = embedder.embed("warmup")

        vector_search = VectorSearch()
        vector_search.load(faiss_path, id_map_path)

        bm25_search = BM25Search()
        bm25_search.lazy_load(bm25_path)  # background thread; first search waits if needed

    hybrid_search = HybridSearch(vector_search, bm25_search, embedder)
    reranker = CrossEncoderReranker(timeout_ms=cross_encoder_timeout_ms)
    scorer = CandidateScorer()

    profiles = ProfileStore()
    offset_idx = faiss_path.parent / "offset_index.json"
    if offset_idx.exists():
        profiles.load_offset_index(offset_idx)
    sample = faiss_path.parent.parent / "samples" / "sample_candidates.json"
    if sample.exists():
        profiles.load_sample(sample)

    planner = PlannerAgent()
    executor = ExecutorAgent(hybrid_search, reranker, scorer, profiles)
    reflector = ReflectorAgent()
    orchestrator = Orchestrator(planner, executor, reflector)

    return orchestrator, vector_search, profiles


def get_llm_client() -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    provider = settings.llm_provider

    if provider == "openai":
        from pydantic import SecretStr
        kwargs: dict[str, Any] = {
            "model": settings.openai_model,
            "api_key": SecretStr(settings.openai_api_key),
            "temperature": 0.1,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
        )
    elif provider == "ollama":
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def check_llm_provider_connected() -> bool:
    """Check if the configured LLM provider is connected and accessible."""
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "openai":
        key = settings.openai_api_key
        if not key or key == "sk-..." or not key.strip():
            return False
        return True
    elif provider == "gemini":
        key = settings.gemini_api_key
        if not key or key == "..." or not key.strip():
            return False
        return True
    elif provider == "ollama":
        import urllib.request
        try:
            # Query base url, e.g. http://localhost:11434/
            # Set timeout to 1.0 seconds so it doesn't hang
            response = urllib.request.urlopen(settings.ollama_base_url, timeout=1.0)
            status: int = response.status
            return status == 200
        except Exception:
            return False
    return False

