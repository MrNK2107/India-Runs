from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/india_runs"
    redis_url: str = "redis://localhost:6379/0"
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    log_level: str = "INFO"
    max_replan_cycles: int = 3
    cross_encoder_timeout_ms: int = 500

    model_config = {"env_file": ".env", "extra": "ignore"}


def load_yaml_config(filename: str) -> dict[str, Any]:
    path = CONFIGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_scoring_config() -> dict[str, Any]:
    return load_yaml_config("scoring_weights.yaml")


@lru_cache
def get_model_config() -> dict[str, Any]:
    return load_yaml_config("models.yaml")


@lru_cache
def get_app_config() -> dict[str, Any]:
    return load_yaml_config("settings.yaml")


@lru_cache
def get_llm_client() -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    provider = settings.llm_provider

    if provider == "openai":
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )
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
