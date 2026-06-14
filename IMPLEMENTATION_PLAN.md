# Implementation Plan — Intelligent Candidate Discovery System
# Every File, Every Function, Every Test. No Shortcuts.

> **This document is the execution blueprint.** Every file lists its exact contents:
> classes, functions, signatures, constants, imports. An agent can follow this
> line-by-line to build the finished product.

---

## Phase 0: Environment Setup

### 0.1 `pyproject.toml`

```toml
[project]
name = "india-runs"
version = "0.1.0"
description = "Intelligent Candidate Discovery System — India Runs Track 1"
requires-python = ">=3.11"
dependencies = [
    # Core
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.9",
    "pydantic-settings>=2.5",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    # ML/NLP
    "sentence-transformers>=3.3",
    "torch>=2.4",
    "transformers>=4.46",
    "langdetect>=1.0.9",
    "spacy>=3.8",
    # Search
    "faiss-cpu>=1.8",
    "rank-bm25>=0.2.2",
    # Agent
    "langgraph>=0.2",
    "langchain-core>=0.3",
    "langchain-openai>=0.2",
    "langchain-google-genai>=2.0",
    "langchain-ollama>=0.2",
    "openai>=1.52",
    "google-genai>=1.0",
    # Data
    "sqlalchemy>=2.0",
    "psycopg2-binary>=2.9",
    "redis>=5.0",
    # UI
    "gradio>=5.0",
    "plotly>=5.24",
    # Dev
    "ruff>=0.7",
    "mypy>=1.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "httpx>=0.27",
    "pytest-cov>=6.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks integration tests",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

### 0.2 `.env.example`

```bash
# LLM Provider (choose one: openai | gemini | ollama)
LLM_PROVIDER=openai

# OpenAI (if provider=openai)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Google Gemini (if provider=gemini)
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash

# Ollama (if provider=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# PostgreSQL
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/india_runs

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# Application
LOG_LEVEL=INFO
MAX_REPLAN_CYCLES=3
CROSS_ENCODER_TIMEOUT_MS=500
```

### 0.3 `docker-compose.yml`

```yaml
version: "3.9"
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: india_runs
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

### 0.4 `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for faiss, psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Build indexes (data must be generated first)
# RUN python scripts/generate_data.py
# RUN python scripts/build_indexes.py

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 0.5 `scripts/deploy.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "=== India Runs — Deployment Script ==="

# 1. Build Docker image
docker compose build

# 2. Start infrastructure
docker compose up -d postgres redis
sleep 5  # Wait for postgres

# 3. Generate synthetic data (if not exists)
if [ ! -f data/profiles/profiles.json ]; then
    echo "Generating synthetic profiles..."
    docker compose run --rm app python scripts/generate_data.py
fi

# 4. Build indexes
echo "Building FAISS + BM25 indexes..."
docker compose run --rm app python scripts/build_indexes.py

# 5. Run evaluation
echo "Running evaluation..."
docker compose run --rm app python scripts/evaluate.py

# 6. Start application
echo "Starting application..."
docker compose up -d app

echo "=== Application running at http://localhost:8000 ==="
echo "=== Gradio UI at http://localhost:7860 ==="
```

### 0.6 Directory structure creation

Run this to create all directories:
```bash
mkdir -p src/{api/{routes,middleware},core,ingestion,language,search,matching,agents,rationale,fairness,data,ui}
mkdir -p tests/{test_ingestion,test_language,test_search,test_matching,test_agents,test_rationale,test_api,test_integration}
mkdir -p notebooks scripts configs data/{profiles,queries,ground_truth,indexes,models} docs
touch src/__init__.py src/{api,api/routes,api/middleware,core,ingestion,language,search,matching,agents,rationale,fairness,data,ui}/__init__.py
touch tests/__init__.py tests/{test_ingestion,test_language,test_search,test_matching,test_agents,test_rationale,test_api,test_integration}/__init__.py
```

---

## Phase 1: Core Infrastructure

### 1.1 `configs/settings.yaml`

```yaml
app:
  name: "india-runs"
  version: "0.1.0"
  log_level: "INFO"

database:
  url: "${DATABASE_URL}"
  pool_size: 10
  max_overflow: 20

redis:
  url: "${REDIS_URL}"
  cache_ttl_seconds: 3600

models:
  embedding:
    name: "paraphrase-multilingual-MiniLM-L12-v2"
    dimension: 384
    max_seq_length: 256
    device: "cpu"
  cross_encoder:
    name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_seq_length: 512
    device: "cpu"
  translation:
    name: "Helsinki-NLP/opus-mt-mul"
    fallback: "facebook/mbart-large-50-many-to-many-mmt"
  planner:
    provider: "${LLM_PROVIDER:-openai}"  # openai | gemini | ollama
    model: "gpt-4o-mini"  # overridden per provider in models.yaml
    temperature: 0.1
  rationale:
    provider: "${LLM_PROVIDER:-openai}"
    model: "gpt-4o-mini"
    temperature: 0.3

search:
  top_k_hybrid: 50
  top_k_final: 10
  rrf_k: 60
  cross_encoder_timeout_ms: 500

scoring:
  weights_file: "configs/scoring_weights.yaml"

agent:
  max_replan_cycles: 3
  min_good_matches_for_pass: 8
```

### 1.2 `configs/scoring_weights.yaml`

```yaml
scoring_weights:
  semantic_similarity: 0.25
  keyword_match: 0.15
  skill_match: 0.30
  experience_match: 0.15
  location_match: 0.05
  education_match: 0.05
  cross_encoder: 0.05

skill_importance_weights:
  required: 1.0
  preferred: 0.6
  nice_to_have: 0.3

proficiency_scores:
  beginner: 0.25
  intermediate: 0.50
  advanced: 0.75
  expert: 1.00

rrf_k: 60
max_replan_cycles: 3
min_good_matches_for_pass: 8
```

### 1.3 `configs/models.yaml`

```yaml
models:
  embedding:
    sentence_transformers_name: "paraphrase-multilingual-MiniLM-L12-v2"
    dimension: 384
    batch_size: 64
    normalize_embeddings: true
  cross_encoder:
    name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: 32
  translation:
    primary: "Helsinki-NLP/opus-mt-mul"
    fallback: "facebook/mbart-large-50-many-to-many-mmt"
  language_detection:
    library: "langdetect"
    seed: 0
```

### 1.4 `src/core/config.py`

```python
"""Application configuration — loads from YAML + env vars."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Main application settings."""

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/india_runs"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # LLM Provider
    llm_provider: str = "openai"  # openai | gemini | ollama
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Application
    log_level: str = "INFO"
    max_replan_cycles: int = 3
    cross_encoder_timeout_ms: int = 500

    model_config = {"env_file": ".env", "extra": "ignore"}


def load_yaml_config(filename: str) -> dict[str, Any]:
    """Load a YAML config file from the configs directory."""
    path = CONFIGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton for application settings."""
    return Settings()


@lru_cache
def get_scoring_config() -> dict[str, Any]:
    """Load scoring weights from YAML."""
    return load_yaml_config("scoring_weights.yaml")


@lru_cache
def get_model_config() -> dict[str, Any]:
    """Load model configurations from YAML."""
    return load_yaml_config("models.yaml")


@lru_cache
def get_app_config() -> dict[str, Any]:
    """Load application settings from YAML."""
    return load_yaml_config("settings.yaml")
```

### 1.5 `src/core/constants.py`

```python
"""Constants and magic numbers used across the application."""

from __future__ import annotations

# Supported languages (ISO 639-1 codes)
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
}

# Source types
PROFILE_SOURCES = ("linkedin", "naukri", "github", "resume_pdf", "career_page", "manual")

# Skill categories
SKILL_CATEGORIES = (
    "programming_language",
    "framework",
    "tool",
    "soft_skill",
    "domain_knowledge",
    "certification",
)

# Proficiency levels
PROFICIENCY_LEVELS = ("beginner", "intermediate", "advanced", "expert")

# Importance levels
SKILL_IMPORTANCE = ("required", "preferred", "nice_to_have")

# Employment types
EMPLOYMENT_TYPES = ("full_time", "part_time", "contract", "freelance", "student")

# Match recommendations
MATCH_RECOMMENDATIONS = ("strong_match", "good_match", "potential_match", "weak_match")

# Search methods
SEARCH_METHODS = ("hybrid", "vector_only", "keyword_only")

# Realistic Indian company names for synthetic data
INDIAN_COMPANIES = [
    "Flipkart", "Zoho", "Freshworks", "TCS", "Infosys", "Wipro", "HCL",
    "Razorpay", "PhonePe", "Swiggy", "Zomato", "Ola", "Paytm", "BYJU'S",
    "PolicyBazaar", "Dream11", "Meesho", "CRED", "Postman", "Hasura",
    "CitrusPay", "Mu Sigma", "Fractal Analytics", "Postman", "BrowserStack",
    "Chargebee", "Zenoti", "InMobi", "BigBasket", "UrbanClap", "Vedantu",
    "Unacademy", "Cult.fit", "Cars24", "Delhivery", "BlackBuck",
    "Icertis", "Druva", "Sigmoid", "Tiger Analytics", "Fractal",
    "Publicis Sapient", "Infosys BPM", "Mindtree", "Mphasis",
]

# Realistic Indian cities
INDIAN_CITIES = [
    "Bangalore", "Hyderabad", "Pune", "Chennai", "Noida", "Gurgaon",
    "Mumbai", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Kochi",
    "Indore", "Bhopal", "Coimbatore", "Visakhapatnam", "Thiruvananthapuram",
    "Mysore", "Nagpur", "Chandigarh",
]

# Indian universities
INDIAN_UNIVERSITIES = [
    "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
    "BITS Pilani", "NIT Trichy", "NIT Warangal", "IIIT Hyderabad",
    "VIT Vellore", "SRM University", "Manipal Institute of Technology",
    "Delhi Technological University", "Punjab Engineering College",
    "Anna University", "Osmania University", "JNTU Hyderabad",
    "University of Mumbai", "Pune University", "Christ University Bangalore",
]

# Hard filter defaults
DEFAULT_MIN_EXPERIENCE = 0
DEFAULT_MAX_EXPERIENCE = 50
DEFAULT_LOCATION = None

# FAISS
FAISS_INDEX_PATH = DATA_DIR / "indexes" / "faiss_index.bin"
FAISS_ID_MAP_PATH = DATA_DIR / "indexes" / "faiss_id_map.json"
BM25_INDEX_PATH = DATA_DIR / "indexes" / "bm25_index.pkl"
PROFILES_PATH = DATA_DIR / "profiles" / "profiles.json"
QUERIES_PATH = DATA_DIR / "queries" / "queries.json"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth" / "ground_truth.json"
```

### 1.6 `src/core/models.py`

> **Full file:** ~350 lines. Contains every Pydantic model from PRD Sections 6.2–6.4.

```python
"""Pydantic models for all data schemas. Single source of truth for validation."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────

class ProfileSource(str, Enum):
    LINKEDIN = "linkedin"
    NAUKRI = "naukri"
    GITHUB = "github"
    RESUME_PDF = "resume_pdf"
    CAREER_PAGE = "career_page"
    MANUAL = "manual"


class SkillCategory(str, Enum):
    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK = "framework"
    TOOL = "tool"
    SOFT_SKILL = "soft_skill"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CERTIFICATION = "certification"


class ProficiencyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SkillImportance(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    NICE_TO_HAVE = "nice_to_have"


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    STUDENT = "student"


class MatchRecommendation(str, Enum):
    STRONG = "strong_match"
    GOOD = "good_match"
    POTENTIAL = "potential_match"
    WEAK = "weak_match"


class SearchMethod(str, Enum):
    HYBRID = "hybrid"
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"


# ── Nested Models (Profile) ───────────────────────────────────

class Location(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    country: str = "India"
    is_remote_ok: bool = False


class PersonalInfo(BaseModel):
    name: str
    location: Location = Field(default_factory=Location)
    languages_spoken: list[str] = Field(default_factory=list)
    native_language: Optional[str] = None


class ProfessionalInfo(BaseModel):
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    total_experience_years: Optional[float] = None
    industry: Optional[str] = None
    employment_type: Optional[EmploymentType] = None


class Skill(BaseModel):
    name: str
    category: SkillCategory = SkillCategory.TOOL
    proficiency: Optional[ProficiencyLevel] = None
    years_used: Optional[float] = None
    evidence: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class WorkExperience(BaseModel):
    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_current: bool = False
    description: str = ""
    highlights: list[str] = Field(default_factory=list)
    skills_demonstrated: list[str] = Field(default_factory=list)
    location: Optional[str] = None


class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    field: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[float] = None


class Signals(BaseModel):
    is_passive: bool = False
    last_active_date: Optional[str] = None
    open_to_work: Optional[bool] = None
    github_activity_score: Optional[float] = None
    has_portfolio: bool = False
    certifications: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)
    speaking_engagements: list[str] = Field(default_factory=list)


class ProfileMetadata(BaseModel):
    language_detected: str = "en"
    original_language: str = "en"
    was_translated: bool = False
    translation_confidence: Optional[float] = None
    embedding_vector_id: Optional[int] = None
    bm25_doc_id: Optional[int] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


# ── Profile (root model) ──────────────────────────────────────

class Profile(BaseModel):
    """Normalized profile schema — PRD Section 6.2."""

    profile_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: ProfileSource = ProfileSource.MANUAL
    raw_text: str = ""
    personal: PersonalInfo
    professional: ProfessionalInfo = Field(default_factory=ProfessionalInfo)
    skills: list[Skill] = Field(default_factory=list)
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    signals: Signals = Field(default_factory=Signals)
    metadata: ProfileMetadata = Field(default_factory=ProfileMetadata)


# ── Job Query ──────────────────────────────────────────────────

class RequiredSkill(BaseModel):
    name: str
    importance: SkillImportance = SkillImportance.REQUIRED
    min_proficiency: Optional[ProficiencyLevel] = None
    min_years: Optional[float] = None


class PreferredSkill(BaseModel):
    name: str
    importance: SkillImportance = SkillImportance.NICE_TO_HAVE
    weight: float = Field(default=0.5, ge=0.0, le=1.0)


class ExperienceRequirements(BaseModel):
    min_years: Optional[float] = None
    max_years: Optional[float] = None
    industry: Optional[str] = None


class LocationRequirements(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    remote_ok: bool = False
    hybrid_ok: bool = False


class EducationRequirements(BaseModel):
    min_degree: Optional[str] = None
    field: Optional[str] = None


class SalaryRequirements(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    currency: str = "INR"


class QueryFilters(BaseModel):
    exclude_companies: list[str] = Field(default_factory=list)
    include_companies: list[str] = Field(default_factory=list)
    must_have_certifications: list[str] = Field(default_factory=list)
    languages_required: list[str] = Field(default_factory=list)


class ParsedQuery(BaseModel):
    """Structured query parsed from natural language — PRD Section 6.3."""

    required_skills: list[RequiredSkill] = Field(default_factory=list)
    preferred_skills: list[PreferredSkill] = Field(default_factory=list)
    experience: ExperienceRequirements = Field(default_factory=ExperienceRequirements)
    location: LocationRequirements = Field(default_factory=LocationRequirements)
    education: EducationRequirements = Field(default_factory=EducationRequirements)
    salary: SalaryRequirements = Field(default_factory=SalaryRequirements)
    filters: QueryFilters = Field(default_factory=QueryFilters)


class JobQuery(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_query: str
    parsed: ParsedQuery = Field(default_factory=ParsedQuery)
    language: str = "en"


# ── Match Results ──────────────────────────────────────────────

class MatchScores(BaseModel):
    """Per-dimension scores for a single match — PRD Section 6.4."""

    overall: float = Field(ge=0.0, le=1.0)
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    keyword_match: float = Field(ge=0.0, le=1.0)
    skill_match: float = Field(ge=0.0, le=1.0)
    experience_match: float = Field(ge=0.0, le=1.0)
    location_match: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    education_match: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cross_encoder_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class SkillDetail(BaseModel):
    skill: str
    required: bool
    found: bool
    proficiency_match: bool
    evidence: str = ""


class Rationale(BaseModel):
    """Per-candidate rationale report — PRD Section 13.1."""

    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    skill_details: list[SkillDetail] = Field(default_factory=list)
    experience_analysis: str = ""
    recommendation: MatchRecommendation = MatchRecommendation.GOOD


class MatchMetadata(BaseModel):
    search_method: SearchMethod = SearchMethod.HYBRID
    reranked: bool = False
    language_matched: bool = False
    passive_candidate: bool = False
    processing_time_ms: int = 0
    translation_fallback: bool = False


class MatchResult(BaseModel):
    """Single candidate match — PRD Section 6.4."""

    match_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str
    profile_id: str
    rank: int
    name: str = ""
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    location: Optional[str] = None
    experience_years: Optional[float] = None
    scores: MatchScores
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    rationale: Rationale = Field(default_factory=Rationale)
    metadata: MatchMetadata = Field(default_factory=MatchMetadata)


# ── API Request/Response Models ────────────────────────────────

class SearchFilters(BaseModel):
    location: Optional[str] = None
    min_experience_years: Optional[float] = None
    max_experience_years: Optional[float] = None
    remote_ok: bool = False
    exclude_companies: list[str] = Field(default_factory=list)
    include_companies: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    """POST /api/v1/search request body — PRD Section 9.1."""

    query: str = Field(min_length=1, max_length=2000)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    max_results: int = Field(default=10, ge=1, le=100)
    include_rationale: bool = True
    language: Optional[str] = None


class SearchResultItem(BaseModel):
    rank: int
    profile_id: str
    name: str
    current_title: Optional[str] = None
    current_company: Optional[str] = None
    location: Optional[str] = None
    experience_years: Optional[float] = None
    scores: MatchScores
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    rationale: Rationale
    passive_candidate: bool = False
    language_matched: bool = False


class SearchMetadata(BaseModel):
    methods_used: list[str] = Field(default_factory=list)
    replan_count: int = 0
    total_time_ms: int = 0


class SearchResponse(BaseModel):
    """POST /api/v1/search response body — PRD Section 9.1."""

    query_id: str
    total_candidates_searched: int
    results: list[SearchResultItem] = Field(default_factory=list)
    message: Optional[str] = None
    suggestions: list[str] = Field(default_factory=list)
    processing_time_ms: int = 0
    search_metadata: SearchMetadata = Field(default_factory=SearchMetadata)


class IngestResponse(BaseModel):
    total_profiles: int
    successful: int
    failed: int
    language_distribution: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    index_size: int
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    last_updated: Optional[str] = None
```

---

## Phase 2: Synthetic Data Generation

### 2.1 `src/data/generator.py`

> **Purpose:** Generate 1,000 realistic Indian candidate profiles + 50 queries + 20 ground-truth sets.

**Classes/Functions:**
```python
class ProfileGenerator:
    """Generate realistic synthetic candidate profiles."""
    
    def __init__(self, seed: int = 42) -> None: ...
    
    def generate_profile(self, language: str = "en") -> dict: ...
    """Generate a single profile with realistic Indian hiring data."""
    
    def generate_batch(self, count: int = 1000) -> list[dict]: ...
    """Generate batch with language distribution: 60% en, 20% hi, 10% ta, 5% te, 5% other."""
    
    def _generate_name(self, language: str) -> str: ...
    """Generate realistic Indian name in the given language."""
    
    def _generate_skills(self, role_type: str) -> list[dict]: ...
    """Generate relevant skills for a given role type."""
    
    def _generate_experience(self, total_years: float) -> list[dict]: ...
    """Generate 2-5 work experiences that sum to total_years."""
    
    def _generate_education(self) -> dict: ...
    """Generate realistic Indian university education."""
    
    def _construct_raw_text(self, profile: dict) -> str: ...
    """Construct raw_text per PRD Section 6.2a."""
    
    def _compute_quality_score(self, profile: dict) -> float: ...
    """Compute data_quality_score per PRD Section 6.2b."""


class QueryGenerator:
    """Generate realistic job search queries."""
    
    def generate_queries(self, count: int = 50) -> list[dict]: ...
    """Generate 50 queries: 15 technical, 15 business, 10 creative, 10 cross-functional."""
    
    def generate_ground_truth(self, queries: list[dict], profiles: list[dict]) -> dict: ...
    """For 20 queries, label top-10 relevant candidates as ground truth."""


def run_generation(output_dir: Path) -> None:
    """Main entry point — generate all data and save to disk."""
```

**Data generation rules (from PRD 6.5-6.6):**
- Each profile: 3+ work experiences, realistic Indian company names
- Language distribution: 60% en, 20% hi, 10% ta, 5% te, 5% other
- 20% non-English names with mixed-language content
- Skills reflect Indian market demand
- Varying quality (some complete, some messy)

### 2.2 `scripts/generate_data.py`

```python
"""CLI script to generate synthetic data."""
from src.data.generator import run_generation
from src.core.constants import DATA_DIR

if __name__ == "__main__":
    run_generation(DATA_DIR)
```

### 2.3 `src/data/ground_truth.py`

```python
"""Ground truth labels for evaluation."""

def load_ground_truth() -> dict[str, list[str]]:
    """Load ground truth from disk. Returns {query_id: [relevant_profile_ids]}."""
    ...

def save_ground_truth(data: dict) -> None:
    """Save ground truth to disk."""
    ...
```

### 2.4 `src/data/sample_queries.py`

```python
"""50 sample queries for demo and evaluation."""

SAMPLE_QUERIES: list[dict] = [
    # Technical (15)
    {"query": "Find a senior DevOps engineer with 5+ years in AWS and Kubernetes, based in Bangalore", "category": "technical"},
    {"query": "我们需要一个有3年以上Python和机器学习经验的后端工程师", "category": "technical"},
    # ... 13 more technical
    
    # Business (15)
    {"query": "Product manager with B2B SaaS experience and growth mindset", "category": "business"},
    # ... 14 more business
    
    # Creative (10)
    {"query": "UX designer who has redesigned enterprise dashboards", "category": "creative"},
    # ... 9 more creative
    
    # Cross-functional (10)
    {"query": "CTO-level leader who has scaled engineering teams from 10 to 100", "category": "crossfunctional"},
    # ... 9 more cross-functional
]
```

---

## Phase 3: Ingestion Pipeline

### 3.1 `src/ingestion/parser.py`

```python
"""Parse raw profiles from JSON/CSV/text into normalized schema."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from src.core.models import Profile

class ProfileParser:
    """Parse profiles from various input formats."""
    
    def parse_json(self, data: dict[str, Any]) -> Profile:
        """Parse a single profile from JSON dict."""
        ...
    
    def parse_json_file(self, path: Path) -> list[Profile]:
        """Parse all profiles from a JSON file."""
        ...
    
    def parse_csv(self, path: Path) -> list[Profile]:
        """Parse profiles from CSV (column-mapped)."""
        ...
    
    def parse_raw_text(self, text: str, source: str = "manual") -> Profile:
        """Parse a raw text resume into a Profile."""
        ...
    
    def parse_batch(self, data: list[dict[str, Any]]) -> tuple[list[Profile], list[str]]:
        """Parse a batch, returning (successful, error_messages)."""
        ...
```

### 3.2 `src/ingestion/extractor.py`

```python
"""LLM-assisted field extraction from unstructured text."""

from __future__ import annotations
from typing import Any

class FieldExtractor:
    """Extract structured fields from unstructured profile text using LLM."""
    
    def __init__(self, model: str = "gpt-4o-mini") -> None: ...
    
    async def extract_skills(self, text: str) -> list[dict[str, Any]]:
        """Extract skills with categories and proficiency from text."""
        ...
    
    async def extract_experience(self, text: str) -> list[dict[str, Any]]:
        """Extract work experience entries from text."""
        ...
    
    async def extract_education(self, text: str) -> list[dict[str, Any]]:
        """Extract education entries from text."""
        ...
    
    async def extract_all(self, text: str) -> dict[str, Any]:
        """Extract all structured fields in one LLM call."""
        ...
    
    def _build_extraction_prompt(self, text: str) -> str:
        """Build the LLM prompt for field extraction."""
        ...
    
    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse and validate LLM JSON response."""
        ...
```

### 3.3 `src/ingestion/normalizer.py`

```python
"""Normalize profiles from different sources to unified schema."""

from __future__ import annotations
from typing import Any
from src.core.models import Profile

class ProfileNormalizer:
    """Normalize profiles across different source formats."""
    
    def normalize_linkedin(self, raw: dict[str, Any]) -> Profile:
        """Normalize LinkedIn export format."""
        ...
    
    def normalize_naukri(self, raw: dict[str, Any]) -> Profile:
        """Normalize Naukri.com scraped format."""
        ...
    
    def normalize_github(self, raw: dict[str, Any]) -> Profile:
        """Normalize GitHub API format."""
        ...
    
    def normalize_generic(self, raw: dict[str, Any]) -> Profile:
        """Generic normalization for unknown formats."""
        ...
    
    def normalize(self, raw: dict[str, Any], source: str) -> Profile:
        """Route to correct normalizer based on source."""
        ...
```

### 3.4 `src/ingestion/quality_scorer.py`

```python
"""Data quality scoring for profiles."""

from __future__ import annotations
import re
from src.core.models import Profile

def has_encoding_artifacts(text: str) -> bool:
    """Check for mojibake or encoding issues."""
    ...

def compute_data_quality_score(profile: Profile) -> float:
    """
    Score 0.0-1.0 based on completeness — PRD Section 6.2b.
    
    Scoring:
    - Name present: +0.10
    - Title present: +0.10
    - At least 1 skill: +0.15
    - At least 1 experience: +0.15
    - Education present: +0.10
    - Location present: +0.10
    - raw_text > 200 chars: +0.10
    - raw_text > 500 chars: +0.05
    - No encoding artifacts: +0.05
    - Skills have evidence: +0.10
    """
    score = 0.0
    if profile.personal.name:
        score += 0.10
    if profile.professional.current_title:
        score += 0.10
    if profile.skills:
        score += 0.15
    if profile.experience:
        score += 0.15
    if profile.education:
        score += 0.10
    if profile.personal.location.city:
        score += 0.10
    raw = profile.raw_text
    if len(raw) > 200:
        score += 0.10
    if len(raw) > 500:
        score += 0.05
    if not has_encoding_artifacts(raw):
        score += 0.05
    if any(s.evidence for s in profile.skills):
        score += 0.10
    return min(score, 1.0)
```

---

## Phase 4: Language Processing

### 4.1 `src/language/detector.py`

```python
"""Language detection for profiles and queries."""

from __future__ import annotations
from langdetect import detect, DetectorFactory, LangDetectException

# Fix seed for reproducibility
DetectorFactory.seed = 0

class LanguageDetector:
    """Detect the language of a text string."""
    
    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence
    
    def detect(self, text: str) -> dict[str, str | bool]:
        """
        Returns: {"language": "hi", "is_english": False, "needs_translation": True}
        Falls back to "en" on detection failure.
        """
        ...
    
    def detect_batch(self, texts: list[str]) -> list[dict[str, str | bool]]:
        """Detect language for a batch of texts."""
        ...
```

### 4.2 `src/language/translator.py`

```python
"""Translation pipeline for non-English profiles."""

from __future__ import annotations
from typing import Optional

class TranslationPipeline:
    """Translate non-English text to English."""
    
    def __init__(self, primary_model: str = "Helsinki-NLP/opus-mt-mul",
                 fallback_model: str = "facebook/mbart-large-50-many-to-many-mmt") -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self._primary_loaded = False
        self._fallback_loaded = False
    
    def load_models(self) -> None:
        """Lazy-load translation models."""
        ...
    
    def translate_to_english(self, text: str, source_lang: str) -> dict[str, str | float]:
        """
        Translate text to English.
        Returns: {"original": ..., "translated": ..., "confidence": 0.95, "model_used": ...}
        """
        ...
    
    def translate_batch(self, texts: list[tuple[str, str]]) -> list[dict]:
        """Translate a batch of (text, source_lang) pairs."""
        ...
    
    def _get_model_name(self, source_lang: str) -> Optional[str]:
        """Get the specific opus-mt model name for a language pair."""
        ...
```

### 4.3 `src/language/multilingual.py`

```python
"""Multilingual embedding utilities."""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

class MultilingualEmbedder:
    """Embed text in 50+ languages into a shared vector space."""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None
        self.dimension: int = 384  # Fixed for MiniLM
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns 384-dim vector."""
        ...
    
    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a batch of texts. Returns (N, 384) array."""
        ...
    
    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        ...
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query (same as embed, but for clarity)."""
        return self.embed(query)
```

---

## Phase 5: Search Engine

### 5.1 `src/search/vector_search.py`

```python
"""FAISS vector similarity search."""

from __future__ import annotations
import json
import faiss
import numpy as np
from pathlib import Path
from src.core.constants import FAISS_INDEX_PATH, FAISS_ID_MAP_PATH

class VectorSearch:
    """FAISS-based dense vector search."""
    
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.index: faiss.IndexFlatIP | None = None
        self.id_map: list[str] = []  # profile_id at each index position
    
    def build_index(self, embeddings: np.ndarray, profile_ids: list[str]) -> None:
        """
        Build FAISS index from embeddings.
        Uses Inner Product (equivalent to cosine on normalized vectors).
        """
        ...
    
    def search(self, query_embedding: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        """
        Search for nearest neighbors.
        Returns: [(profile_id, score), ...] sorted by score descending.
        """
        ...
    
    def save(self, index_path: Path = FAISS_INDEX_PATH, 
             id_map_path: Path = FAISS_ID_MAP_PATH) -> None:
        """Persist index and ID map to disk."""
        ...
    
    def load(self, index_path: Path = FAISS_INDEX_PATH,
             id_map_path: Path = FAISS_ID_MAP_PATH) -> None:
        """Load index and ID map from disk."""
        ...
    
    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal if self.index else 0
```

### 5.2 `src/search/bm25_search.py`

```python
"""BM25 keyword search."""

from __future__ import annotations
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from src.core.constants import BM25_INDEX_PATH

class BM25Search:
    """BM25-based sparse keyword search."""
    
    def __init__(self) -> None:
        self.index: BM25Okapi | None = None
        self.id_map: list[str] = []
        self.corpus_tokenized: list[list[str]] = []
    
    def build_index(self, documents: list[str], profile_ids: list[str]) -> None:
        """
        Build BM25 index from tokenized documents.
        Tokenization: lowercase + simple whitespace split.
        """
        ...
    
    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """
        Search for documents matching query.
        Returns: [(profile_id, bm25_score), ...] sorted by score descending.
        """
        ...
    
    def save(self, path: Path = BM25_INDEX_PATH) -> None:
        """Persist index to disk."""
        ...
    
    def load(self, path: Path = BM25_INDEX_PATH) -> None:
        """Load index from disk."""
        ...
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase + whitespace split."""
        ...
    
    @property
    def size(self) -> int:
        return len(self.id_map)
```

### 5.3 `src/search/hybrid.py`

```python
"""Hybrid search orchestrator — BM25 + FAISS + RRF."""

from __future__ import annotations
import numpy as np
from src.search.vector_search import VectorSearch
from src.search.bm25_search import BM25Search
from src.language.multilingual import MultilingualEmbedder
from src.core.config import get_scoring_config

class HybridSearch:
    """Orchestrate parallel vector + keyword search with RRF fusion."""
    
    def __init__(self, vector_search: VectorSearch, bm25_search: BM25Search,
                 embedder: MultilingualEmbedder) -> None:
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.embedder = embedder
        self.rrf_k = get_scoring_config().get("rrf_k", 60)
    
    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """
        Execute hybrid search:
        1. Embed query
        2. Search FAISS (vector search)
        3. Search BM25 (keyword search)
        4. Fuse with Reciprocal Rank Fusion
        Returns: [(profile_id, rrf_score), ...] sorted by score descending.
        """
        ...
    
    def reciprocal_rank_fusion(
        self, 
        rankings: list[list[tuple[str, float]]],
        k: int = 60
    ) -> list[tuple[str, float]]:
        """
        Combine multiple ranked lists using RRF.
        RRF_score(d) = Σ 1/(k + rank_i(d))
        """
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 5.4 `src/search/reranker.py`

```python
"""Cross-encoder reranking."""

from __future__ import annotations
import time
from typing import Optional
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Rerank search results using a cross-encoder model."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 timeout_ms: int = 500) -> None:
        self.model_name = model_name
        self.timeout_ms = timeout_ms
        self._model: CrossEncoder | None = None
    
    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model
    
    def rerank(self, query: str, 
               candidates: list[tuple[str, str, float]],  # (id, text, original_score)
               top_k: int = 10) -> list[tuple[str, float]]:
        """
        Rerank candidates using cross-encoder.
        Returns: [(profile_id, cross_encoder_score), ...] sorted by score.
        Times out and returns original ranking if exceeds timeout.
        """
        ...
    
    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        ...
```

### 5.5 `src/search/filters.py`

```python
"""Hard filter application for search results."""

from __future__ import annotations
from typing import Optional
from src.core.models import Profile, SearchFilters

class SearchFilter:
    """Apply hard filters to candidate profiles before or after search."""
    
    def __init__(self, filters: SearchFilters) -> None:
        self.filters = filters
    
    def passes(self, profile: Profile) -> bool:
        """Check if a profile passes all hard filters."""
        ...
    
    def filter_profiles(self, profiles: list[Profile]) -> list[Profile]:
        """Filter a list of profiles."""
        ...
    
    def _check_location(self, profile: Profile) -> bool:
        """Check location filter."""
        ...
    
    def _check_experience(self, profile: Profile) -> bool:
        """Check experience range filter."""
        ...
    
    def _check_companies(self, profile: Profile) -> bool:
        """Check company include/exclude filters."""
        ...
```

---

## Phase 6: Matching & Scoring

### 6.1 `src/matching/skill_matcher.py`

```python
"""Skill matching with fuzzy logic."""

from __future__ import annotations
from difflib import SequenceMatcher
from typing import Optional
from src.core.models import Skill, RequiredSkill, SkillImportance

class SkillMatcher:
    """Match required skills against candidate skills with fuzzy matching."""
    
    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self.similarity_threshold = similarity_threshold
    
    def match_skills(self, required: list[RequiredSkill], 
                     candidate_skills: list[Skill]) -> tuple[float, list[dict]]:
        """
        Compute skill match score.
        Returns: (score, skill_details_list)
        """
        ...
    
    def find_best_match(self, required_name: str, 
                        candidate_skills: list[Skill]) -> Optional[Skill]:
        """
        Find best matching candidate skill using:
        1. Exact match (case-insensitive)
        2. Normalized string match (strip, lowercase)
        3. Fuzzy match (SequenceMatcher ratio)
        """
        ...
    
    def _normalize(self, name: str) -> str:
        """Normalize skill name for comparison."""
        ...
    
    def _fuzzy_score(self, a: str, b: str) -> float:
        """Compute fuzzy similarity between two skill names."""
        ...
    
    def compute_proficiency_match(self, required: Optional[str], 
                                  candidate: Optional[str]) -> float:
        """Compute proficiency match score."""
        ...

# Common skill aliases for fuzzy matching
SKILL_ALIASES: dict[str, list[str]] = {
    "python": ["python3", "py"],
    "javascript": ["js", "ecmascript", "es6"],
    "react": ["reactjs", "react.js"],
    "kubernetes": ["k8s"],
    "amazon web services": ["aws"],
    "google cloud platform": ["gcp", "google cloud"],
    "microsoft azure": ["azure"],
    "machine learning": ["ml"],
    "artificial intelligence": ["ai"],
    "natural language processing": ["nlp"],
    "continuous integration": ["ci"],
    "continuous deployment": ["cd", "ci/cd"],
    # ... 50+ more
}
```

### 6.2 `src/matching/experience_matcher.py`

```python
"""Experience scoring."""

from __future__ import annotations
from typing import Optional

class ExperienceMatcher:
    """Score how well a candidate's experience matches requirements."""
    
    def match(self, required_min_years: Optional[float],
              required_max_years: Optional[float],
              candidate_years: Optional[float],
              required_industry: Optional[str],
              candidate_industry: Optional[str]) -> float:
        """
        Score experience match (0.0 - 1.0).
        Components: years match + industry match.
        """
        ...
    
    def _score_years(self, required_min: Optional[float],
                     required_max: Optional[float],
                     candidate: Optional[float]) -> Optional[float]:
        """Score years of experience."""
        ...
    
    def _score_industry(self, required: Optional[str],
                        candidate: Optional[str]) -> Optional[float]:
        """Score industry relevance."""
        ...
```

### 6.3 `src/matching/scorer.py`

```python
"""Overall scoring — weighted combination of all dimensions."""

from __future__ import annotations
from src.core.models import MatchScores
from src.core.config import get_scoring_config

class CandidateScorer:
    """Compute overall match score from per-dimension scores."""
    
    def __init__(self) -> None:
        config = get_scoring_config()
        self.weights = config["scoring_weights"]
    
    def compute_overall(self, scores: dict[str, float | None]) -> MatchScores:
        """
        Compute overall score as weighted combination.
        overall = Σ weight_i * score_i (skipping null components, re-normalizing).
        """
        ...
    
    def compute_confidence(self, scores: dict[str, float | None]) -> float:
        """
        Confidence = 1.0 - std_dev of non-null scores.
        High confidence = all dimensions agree.
        """
        ...
```

### 6.4 `src/matching/confidence.py`

```python
"""Confidence calculation utilities."""

from __future__ import annotations
import numpy as np

def compute_score_variance(scores: list[float]) -> float:
    """Compute variance of score list."""
    ...

def compute_confidence(scores: dict[str, float | None]) -> float:
    """
    High confidence when all non-null dimensions agree.
    confidence = 1.0 - normalized_std_dev
    """
    ...
```

---

## Phase 7: Agentic Workflow

### 7.1 `src/agents/prompts.py`

```python
"""All agent system prompts — single source of truth."""

PLANNER_SYSTEM_PROMPT = """You are an expert recruiter's assistant. Given a natural language job query, 
parse it into a structured search specification.

Extract:
- Required skills (with importance: required/preferred/nice_to_have)
- Experience requirements (years, industry)
- Location preferences (city, remote preference)
- Education requirements
- Any exclusion criteria

Output valid JSON matching this schema:
{
  "required_skills": [{"name": "...", "importance": "required|preferred|nice_to_have", "min_proficiency": "...", "min_years": null}],
  "preferred_skills": [{"name": "...", "importance": "nice_to_have", "weight": 0.5}],
  "experience": {"min_years": null, "max_years": null, "industry": null},
  "location": {"city": null, "state": null, "country": null, "remote_ok": false, "hybrid_ok": false},
  "education": {"min_degree": null, "field": null},
  "filters": {"exclude_companies": [], "include_companies": [], "must_have_certifications": [], "languages_required": []}
}

If the query is ambiguous, make reasonable assumptions and note them.
Output ONLY valid JSON, no other text."""

REFLECTOR_SYSTEM_PROMPT = """You are a critical hiring evaluator. For each candidate in the search results,
assess whether they truly match the job requirements.

For each candidate, provide:
1. overall_assessment: "strong_match" | "good_match" | "potential_match" | "weak_match"
2. key_strengths: list of specific reasons why they match
3. key_gaps: list of specific reasons why they might not match
4. concerns: any red flags or uncertainties
5. should_keep: boolean

Be strict — a "strong_match" means you would confidently shortlist this person.
A "good_match" means they could work with some caveats.
A "potential_match" means they're worth a phone screen.
A "weak_match" means they should be dropped.

Output valid JSON array. No other text."""

RATIONALE_SYSTEM_PROMPT = """You are generating a candidate evaluation report for a recruiter.

JOB REQUIREMENTS:
{job_requirements_json}

CANDIDATE PROFILE:
{candidate_profile_summary}

MATCH SCORES:
{scores_json}

Generate a detailed rationale report. Be specific — reference actual companies,
roles, and skills from the profile. Do not make generic statements.

Requirements:
- summary: 2-3 sentences, specific to this candidate
- strengths: list specific matches with evidence
- gaps: list specific concerns or missing requirements
- skill_details: for each required skill, note if found and the evidence
- experience_analysis: paragraph about work history relevance
- recommendation: one of strong_match, good_match, potential_match, weak_match

Output valid JSON only."""

REPLAN_SYSTEM_PROMPT = """The previous search did not yield enough strong matches.
Original query: {original_query}
Previous parsed parameters: {previous_params}
Reflector feedback: {feedback}

Revise the search parameters based on the feedback.
Common revisions:
- Broaden skill requirements (move some required to preferred)
- Relax experience requirements
- Expand location (remove city filter, allow remote)
- Remove company exclusions

Output the revised parsed query as valid JSON (same schema as before)."""
```

### 7.2 `src/agents/planner.py`

```python
"""Planner agent — parse natural language into structured query."""

from __future__ import annotations
import json
import logging
from typing import Any
from src.core.config import get_settings, get_llm_client
from src.core.models import ParsedQuery
from src.agents.prompts import PLANNER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class PlannerAgent:
    """Parse natural language query into structured search parameters."""
    
    def __init__(self) -> None:
        settings = get_settings()
        self.client = get_llm_client()
        self.model = settings.planner_model
    
    async def plan(self, raw_query: str) -> ParsedQuery:
        """
        Parse a natural language job query into structured parameters.
        Falls back to keyword extraction if LLM unavailable.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_query},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content)
            return ParsedQuery(**parsed)
        except Exception as e:
            logger.warning(f"Planner LLM failed, using fallback: {e}")
            return self._fallback_parse(raw_query)
    
    async def replan(self, original_query: str, previous_params: dict,
                     feedback: str) -> ParsedQuery:
        """Re-plan with reflector feedback."""
        ...
    
    def _fallback_parse(self, query: str) -> ParsedQuery:
        """Extract keywords using spaCy NER as fallback."""
        ...
```

### 7.3 `src/agents/executor.py`

```python
"""Executor agent — run hybrid search with parsed parameters."""

from __future__ import annotations
import logging
from src.core.models import ParsedQuery, MatchResult
from src.search.hybrid import HybridSearch
from src.search.reranker import CrossEncoderReranker
from src.search.filters import SearchFilter
from src.matching.scorer import CandidateScorer
from src.search.filters import SearchFilters

logger = logging.getLogger(__name__)

class ExecutorAgent:
    """Execute search using parsed query parameters."""
    
    def __init__(self, hybrid_search: HybridSearch, 
                 reranker: CrossEncoderReranker,
                 scorer: CandidateScorer,
                 profiles: dict) -> None:  # profile_id -> Profile
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.scorer = scorer
        self.profiles = profiles
    
    async def execute(self, parsed: ParsedQuery, 
                      top_k: int = 50) -> list[MatchResult]:
        """
        1. Convert parsed query to search text
        2. Run hybrid search
        3. Apply hard filters
        4. Rerank top-50
        5. Score top-10
        Returns: sorted list of MatchResult
        """
        ...
    
    def _query_to_search_text(self, parsed: ParsedQuery) -> str:
        """Convert structured query back to searchable text."""
        ...
    
    def _apply_filters(self, results: list[tuple[str, float]], 
                       parsed: ParsedQuery) -> list[tuple[str, float]]:
        """Apply hard filters from parsed query."""
        ...
```

### 7.4 `src/agents/reflector.py`

```python
"""Reflector agent — evaluate search results quality."""

from __future__ import annotations
import json
import logging
from typing import Any
from src.core.config import get_settings, get_llm_client
from src.core.models import MatchResult, ParsedQuery
from src.agents.prompts import REFLECTOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class ReflectorAgent:
    """Evaluate search results and decide whether to re-plan."""
    
    def __init__(self) -> None:
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    
    async def reflect(self, query: ParsedQuery, 
                      results: list[MatchResult]) -> dict[str, Any]:
        """
        Evaluate each candidate match.
        Returns: {
            "evaluations": [{"profile_id": ..., "assessment": ..., "should_keep": bool, ...}],
            "good_match_count": int,
            "should_replan": bool,
            "feedback": str
        }
        """
        ...
    
    def _should_replan(self, evaluations: list[dict], 
                       threshold: int = 8) -> bool:
        """Decide if re-plan is needed (less than threshold good matches)."""
        ...
```

### 7.5 `src/agents/orchestrator.py`

```python
"""LangGraph state machine — Plan → Execute → Reflect → Re-plan."""

from __future__ import annotations
import logging
from typing import Any, TypedDict
from langgraph.graph import StateGraph, END
from src.agents.planner import PlannerAgent
from src.agents.executor import ExecutorAgent
from src.agents.reflector import ReflectorAgent
from src.core.models import ParsedQuery, MatchResult, SearchResponse
from src.core.config import get_scoring_config

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    raw_query: str
    parsed_query: dict | None
    results: list[dict]
    evaluations: dict | None
    replan_count: int
    max_replans: int
    should_continue: bool
    search_metadata: dict

class Orchestrator:
    """Main agentic orchestrator — Plan → Execute → Reflect → Re-plan."""
    
    def __init__(self, planner: PlannerAgent, executor: ExecutorAgent,
                 reflector: ReflectorAgent) -> None:
        self.planner = planner
        self.executor = executor
        self.reflector = reflector
        config = get_scoring_config()
        self.max_replans = config.get("max_replan_cycles", 3)
        self.min_good_matches = config.get("min_good_matches_for_pass", 8)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("generate_rationale", self._rationale_node)
        
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "execute")
        workflow.add_edge("execute", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue,
            {
                "replan": "plan",
                "done": "generate_rationale",
            }
        )
        workflow.add_edge("generate_rationale", END)
        return workflow.compile()
    
    async def run(self, raw_query: str) -> SearchResponse:
        """Execute the full agentic workflow."""
        initial_state: AgentState = {
            "raw_query": raw_query,
            "parsed_query": None,
            "results": [],
            "evaluations": None,
            "replan_count": 0,
            "max_replans": self.max_replans,
            "should_continue": True,
            "search_metadata": {},
        }
        final_state = await self.graph.ainvoke(initial_state)
        return self._build_response(final_state)
    
    async def _plan_node(self, state: AgentState) -> dict:
        """Parse query into structured parameters."""
        ...
    
    async def _execute_node(self, state: AgentState) -> dict:
        """Run hybrid search + reranking."""
        ...
    
    async def _reflect_node(self, state: AgentState) -> dict:
        """Evaluate results quality."""
        ...
    
    async def _rationale_node(self, state: AgentState) -> dict:
        """Generate rationale for top matches."""
        ...
    
    def _should_continue(self, state: AgentState) -> str:
        """Decide: replan or done."""
        ...
    
    def _build_response(self, state: AgentState) -> SearchResponse:
        """Convert final state to API response."""
        ...
```

---

## Phase 8: Rationale Generation

### 8.1 `src/rationale/generator.py`

```python
"""Rationale generation for each candidate match."""

from __future__ import annotations
import json
import logging
from openai import AsyncOpenAI
from src.core.config import get_settings
from src.core.models import Rationale, MatchResult, Profile, SkillDetail, MatchRecommendation
from src.agents.prompts import RATIONALE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class RationaleGenerator:
    """Generate detailed rationale reports for each candidate match."""
    
    def __init__(self) -> None:
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    
    async def generate(self, match: MatchResult, profile: Profile,
                       job_requirements: dict) -> Rationale:
        """
        Generate a rationale report for a single candidate match.
        Falls back to template-based rationale if LLM unavailable.
        """
        ...
    
    async def generate_batch(self, matches: list[MatchResult],
                             profiles: dict[str, Profile],
                             job_requirements: dict) -> list[Rationale]:
        """Generate rationale for all top matches."""
        ...
    
    def _build_prompt(self, match: MatchResult, profile: Profile,
                      job_requirements: dict) -> str:
        """Build the rationale generation prompt."""
        ...
    
    def _parse_response(self, response: str) -> Rationale:
        """Parse LLM response into Rationale model."""
        ...
    
    def _template_rationale(self, match: MatchResult, profile: Profile) -> Rationale:
        """Fallback template-based rationale when LLM is unavailable."""
        ...
```

### 8.2 `src/rationale/templates.py`

```python
"""Rationale prompt templates."""

RATIONALE_TEMPLATE = """Generate a candidate evaluation report for a recruiter.

JOB REQUIREMENTS:
{job_requirements}

CANDIDATE PROFILE:
Name: {name}
Title: {current_title} at {current_company}
Experience: {experience_years} years
Skills: {skill_names}
Location: {location}

MATCH SCORES:
- Overall: {overall_score:.2f}
- Skill Match: {skill_score:.2f}
- Experience Match: {experience_score:.2f}
- Semantic Match: {semantic_score:.2f}

Generate a JSON response with:
- summary: 2-3 sentence overview
- strengths: list of specific strengths with evidence
- gaps: list of specific gaps
- skill_details: for each required skill, note found/evidence
- experience_analysis: paragraph about work history relevance
- recommendation: strong_match | good_match | potential_match | weak_match"""

SKILL_EVIDENCE_TEMPLATE = """Candidate skill: {skill_name} (proficiency: {proficiency})
Evidence: {evidence}
Required level: {required_level}
Match: {matched}"""
```

### 8.3 `src/rationale/validator.py`

```python
"""Validate rationale quality."""

from __future__ import annotations
from src.core.models import Rationale

class RationaleValidator:
    """Validate that generated rationales meet quality standards."""
    
    def validate(self, rationale: Rationale) -> tuple[bool, list[str]]:
        """
        Validate a rationale.
        Returns: (is_valid, list of issues)
        
        Checks:
        - summary is 10-500 chars
        - strengths has 1+ items
        - gaps can be empty
        - recommendation is valid enum
        - skill_details covers all required skills
        """
        ...
    
    def validate_batch(self, rationales: list[Rationale]) -> dict[str, int]:
        """Validate a batch, return stats."""
        ...
```

---

## Phase 9: Fairness & Bias

### 9.1 `src/fairness/bias_detector.py`

```python
"""Bias detection utilities."""

from __future__ import annotations
from typing import Optional
from src.core.models import MatchResult, Profile

class BiasDetector:
    """Detect potential bias in match results."""
    
    def check_name_bias(self, matches: list[MatchResult], 
                        profiles: dict[str, Profile]) -> dict[str, Any]:
        """
        Check if certain name patterns are systematically ranked lower.
        Returns bias indicators.
        """
        ...
    
    def check_language_bias(self, matches: list[MatchResult],
                            profiles: dict[str, Profile]) -> dict[str, Any]:
        """Check if non-English profiles are systematically disadvantaged."""
        ...
    
    def check_location_bias(self, matches: list[MatchResult],
                            profiles: dict[str, Profile]) -> dict[str, Any]:
        """Check if tier-2/3 city candidates are disadvantaged."""
        ...
    
    def check_university_bias(self, matches: list[MatchResult],
                              profiles: dict[str, Profile]) -> dict[str, Any]:
        """Check if certain universities are overrepresented in top results."""
        ...
```

### 9.2 `src/fairness/metrics.py`

```python
"""Fairness metrics computation."""

from __future__ import annotations
import numpy as np
from src.core.models import MatchResult, Profile

def compute_demographic_parity(matches: list[MatchResult],
                               profiles: dict[str, Profile],
                               protected_attribute: str) -> float:
    """
    Compute demographic parity for a protected attribute.
    Should be close to 1.0 for fairness.
    """
    ...

def compute_disparate_impact_ratio(matches: list[MatchResult],
                                   profiles: dict[str, Profile],
                                   protected_group: str,
                                   majority_group: str) -> float:
    """
    4/5ths rule: selection_rate_protected / selection_rate_majority.
    Should be ≥ 0.80.
    """
    ...

def compute_language_bias(matches: list[MatchResult],
                          profiles: dict[str, Profile]) -> dict[str, float]:
    """Check if non-English profiles have lower average ranks."""
    ...

def compute_location_bias(matches: list[MatchResult],
                          profiles: dict[str, Profile]) -> dict[str, float]:
    """Check if tier-2/3 city candidates have lower average ranks."""
    ...

def compute_all_fairness_metrics(matches: list[MatchResult],
                                 profiles: dict[str, Profile]) -> dict[str, Any]:
    """Compute all fairness metrics."""
    ...
```

---

## Phase 10: API Layer

### 10.1 `src/main.py`

```python
"""FastAPI application entry point."""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routes.search import router as search_router
from src.api.routes.profiles import router as profiles_router
from src.api.routes.ingest import router as ingest_router
from src.api.routes.health import router as health_router
from src.api.middleware.logging import RequestLoggingMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Starting India Runs — Intelligent Candidate Discovery")
    # Load indexes, models, etc.
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
```

### 10.2 `src/api/routes/search.py`

```python
"""POST /api/v1/search endpoint."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from src.core.models import SearchRequest, SearchResponse
from src.agents.orchestrator import Orchestrator

router = APIRouter()

# Global orchestrator instance (initialized at startup)
_orchestrator: Orchestrator | None = None

@router.post("/search", response_model=SearchResponse)
async def search_candidates(request: SearchRequest) -> SearchResponse:
    """
    Search for candidates matching a natural language query.
    
    - **query**: Natural language job description
    - **filters**: Optional hard filters (location, experience, etc.)
    - **max_results**: Number of results (1-100, default 10)
    - **include_rationale**: Whether to include rationale reports
    """
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        response = await _orchestrator.run(request.query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
```

### 10.3 `src/api/routes/profiles.py`

```python
"""GET /api/v1/profiles/{profile_id} endpoint."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from src.core.models import Profile

router = APIRouter()

@router.get("/profiles/{profile_id}", response_model=Profile)
async def get_profile(profile_id: str) -> Profile:
    """Get full normalized profile data by ID."""
    ...
```

### 10.4 `src/api/routes/ingest.py`

```python
"""POST /api/v1/ingest endpoint."""

from __future__ import annotations
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.core.models import IngestResponse

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_profiles(file: UploadFile = File(...)) -> IngestResponse:
    """
    Bulk ingest profiles from uploaded JSON file.
    Returns ingestion report with counts and language distribution.
    """
    ...
```

### 10.5 `src/api/routes/health.py`

```python
"""GET /api/v1/health endpoint."""

from __future__ import annotations
from fastapi import APIRouter
from src.core.models import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    System health status.
    Returns index sizes, model loading status, last update time.
    """
    ...
```

### 10.6 `src/api/middleware/logging.py`

```python
"""Request/response logging middleware."""

from __future__ import annotations
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("api.access")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests with timing."""
    
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration={duration:.1f}ms"
        )
        return response
```

### 10.7 `src/api/middleware/validation.py`

```python
"""Input validation middleware."""

from __future__ import annotations
from starlette.middleware.base import BaseHTTPMiddleware

class InputValidationMiddleware(BaseHTTPMiddleware):
    """Validate request inputs (max body size, content type, etc.)."""
    
    MAX_BODY_SIZE = 10 * 1024 * 1024  # 10MB
    
    async def dispatch(self, request, call_next):
        ...
```

---

## Phase 11: UI Layer

### 11.1 `src/ui/app.py`

> **This is the main Gradio application. ~400 lines.**

```python
"""Gradio demo application — the primary user interface."""

from __future__ import annotations
import gradio as gr
import plotly.graph_objects as go
from src.core.models import SearchResponse, SearchResultItem
from src.ui.components import (
    create_candidate_card,
    create_score_radar_chart,
    create_skill_match_table,
    create_analytics_dashboard,
    MATCH_COLORS,
)

def create_app() -> gr.Blocks:
    """Create the main Gradio application."""
    
    with gr.Blocks(
        title="India Runs — Intelligent Candidate Discovery",
        theme=gr.themes.Soft(),
        css="src/ui/styles.css",
    ) as app:
        
        gr.Markdown("# 🏃 India Runs — Intelligent Candidate Discovery")
        gr.Markdown("*Beyond keywords. Beyond filters. AI that understands hiring.*")
        
        with gr.Tabs():
            # Tab 1: Search
            with gr.Tab("🔍 Search"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Job Query",
                            placeholder="e.g., Find a senior DevOps engineer with 5+ years in AWS...",
                            lines=3,
                        )
                        # Example queries
                        gr.Examples(
                            examples=[
                                "Find a senior Python developer with ML experience in Bangalore",
                                "पायथन और डेटा स�ंस में 3 साल का अनुभव वाला उम्मीदवार ढूंढें",
                                "Someone who can build our recommendation engine from scratch",
                                "Product manager with B2B SaaS experience and growth mindset",
                            ],
                            inputs=query_input,
                        )
                    with gr.Column(scale=1):
                        location_filter = gr.Textbox(label="Location")
                        experience_filter = gr.Slider(
                            label="Min Experience (years)", minimum=0, maximum=20, step=1, value=0
                        )
                        remote_ok = gr.Checkbox(label="Remote OK", value=False)
                
                search_btn = gr.Button("🔍 Search Candidates", variant="primary", size="lg")
                results_area = gr.HTML(label="Results")
                rationale_area = gr.HTML(label="Rationale Report")
            
            # Tab 2: Analytics
            with gr.Tab("📊 Analytics"):
                analytics_html = gr.HTML(label="Analytics Dashboard")
            
            # Tab 3: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About This System
                
                **Intelligent Candidate Discovery** — a hybrid semantic search system
                that goes beyond keyword matching.
                
                ### Architecture
                - **Hybrid Search**: BM25 + FAISS vector search + Reciprocal Rank Fusion
                - **Cross-Encoder Reranking**: MiniLM for precision
                - **Agentic Workflow**: Plan → Execute → Reflect → Re-plan (LangGraph)
                - **Multilingual**: 30+ Indian languages via multilingual embeddings
                - **Rationale Reports**: Every match comes with an explanation
                
                ### Tech Stack
                - FastAPI, FAISS, sentence-transformers, LangGraph, Gradio
                """)
        
        # Wire up search
        search_btn.click(
            fn=search_handler,
            inputs=[query_input, location_filter, experience_filter, remote_ok],
            outputs=[results_area, rationale_area],
        )
    
    return app


def search_handler(query, location, min_experience, remote_ok) -> tuple[str, str]:
    """Handle search request and return HTML for results + rationale."""
    # This calls the orchestrator
    ...


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
```

### 11.2 `src/ui/components.py`

```python
"""Reusable UI components for Gradio."""

from __future__ import annotations
from src.core.models import SearchResultItem, Rationale

MATCH_COLORS = {
    "strong_match": "#10b981",  # Green
    "good_match": "#3b82f6",    # Blue
    "potential_match": "#f59e0b",  # Yellow
    "weak_match": "#ef4444",    # Red
}

def create_candidate_card(item: SearchResultItem) -> str:
    """Generate HTML card for a single candidate."""
    ...

def create_score_radar_chart(scores: dict) -> go.Figure:
    """Generate Plotly radar chart for score breakdown."""
    ...

def create_skill_match_table(rationale: Rationale) -> str:
    """Generate HTML table showing skill-by-skill match details."""
    ...

def create_analytics_dashboard(matches_data: list) -> str:
    """Generate analytics dashboard HTML with charts."""
    ...

def create_rationale_panel(rationale: Rationale, profile_summary: str) -> str:
    """Generate full rationale report panel."""
    ...

def create_loading_spinner() -> str:
    """HTML/CSS loading animation."""
    ...
```

### 11.3 `src/ui/styles.css`

```css
/* Custom CSS for professional look */

/* Score badges */
.score-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 14px;
}
.score-strong { background: #d1fae5; color: #065f46; }
.score-good { background: #dbeafe; color: #1e40af; }
.score-potential { background: #fef3c7; color: #92400e; }
.score-weak { background: #fee2e2; color: #991b1b; }

/* Skill chips */
.skill-chip {
    display: inline-block;
    padding: 2px 8px;
    margin: 2px;
    border-radius: 4px;
    font-size: 12px;
    background: #f3f4f6;
    color: #374151;
}
.skill-chip.matched { background: #d1fae5; color: #065f46; }
.skill-chip.missing { background: #fee2e2; color: #991b1b; }

/* Candidate card */
.candidate-card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    transition: box-shadow 0.2s;
}
.candidate-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* Radar chart container */
.radar-container {
    width: 300px;
    height: 300px;
}
```

---

## Phase 12: Index Building

### 12.1 `scripts/build_indexes.py`

```python
"""Build FAISS + BM25 indexes from generated data."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from src.core.constants import PROFILES_PATH, FAISS_INDEX_PATH, BM25_INDEX_PATH
from src.language.multilingual import MultilingualEmbedder
from src.search.vector_search import VectorSearch
from src.search.bm25_search import BM25Search
from src.ingestion.quality_scorer import compute_data_quality_score
from src.core.models import Profile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_indexes() -> None:
    """Main entry point — build all search indexes."""
    # 1. Load profiles
    profiles = load_profiles(PROFILES_PATH)
    logger.info(f"Loaded {len(profiles)} profiles")
    
    # 2. Build embedding index
    embedder = MultilingualEmbedder()
    raw_texts = [p.raw_text for p in profiles]
    profile_ids = [p.profile_id for p in profiles]
    
    logger.info("Generating embeddings...")
    embeddings = embedder.embed_batch(raw_texts)
    
    vector_search = VectorSearch(dimension=384)
    vector_search.build_index(embeddings, profile_ids)
    vector_search.save()
    logger.info(f"FAISS index built: {vector_search.size} vectors")
    
    # 3. Build BM25 index
    bm25_search = BM25Search()
    bm25_search.build_index(raw_texts, profile_ids)
    bm25_search.save()
    logger.info(f"BM25 index built: {bm25_search.size} documents")
    
    logger.info("All indexes built successfully!")

def load_profiles(path: Path) -> list[Profile]:
    """Load profiles from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return [Profile(**p) for p in data]

if __name__ == "__main__":
    build_indexes()
```

---

## Phase 13: Evaluation

### 13.1 `scripts/evaluate.py`

```python
"""Run full evaluation metrics on test queries."""

from __future__ import annotations
import json
import time
import logging
from statistics import mean, median
from pathlib import Path
from src.core.constants import QUERIES_PATH, GROUND_TRUTH_PATH, PROFILES_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate():
    """Run full evaluation."""
    ...

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Precision@k."""
    ...

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute Recall@k."""
    ...

def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Compute MRR."""
    ...

def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Compute nDCG@k."""
    ...

def cross_lingual_mrr(results: dict) -> float:
    """Compute MRR for non-English queries only."""
    ...

def latency_stats(latencies: list[float]) -> dict:
    """Compute p50, p95, p99 latency."""
    ...

if __name__ == "__main__":
    evaluate()
```

---

## Phase 14: Testing

### 14.1 `tests/conftest.py`

```python
"""Shared pytest fixtures."""

import pytest
from src.core.models import Profile, Skill, WorkExperience, PersonalInfo, ProfessionalInfo

@pytest.fixture
def sample_profile() -> Profile:
    """Create a sample profile for testing."""
    ...

@pytest.fixture
def sample_profiles() -> list[Profile]:
    """Create 20 sample profiles for testing."""
    ...

@pytest.fixture
def sample_query():
    """Create a sample job query."""
    ...

@pytest.fixture
def multilingual_embedder():
    """Create a loaded MultilingualEmbedder."""
    ...

@pytest.fixture
def vector_search():
    """Create an empty VectorSearch."""
    ...

@pytest.fixture
def bm25_search():
    """Create an empty BM25Search."""
    ...
```

### 14.2 Test Files (one per module)

Each test file follows the same pattern — see PRD Section 20.2 for the exact test cases.

| Test File | Tests | What It Verifies |
|-----------|-------|-----------------|
| `test_ingestion/test_parser.py` | 8 tests | JSON parsing, CSV parsing, raw text parsing, batch parsing, error handling |
| `test_ingestion/test_normalizer.py` | 6 tests | LinkedIn, Naukri, GitHub normalization, generic fallback |
| `test_language/test_detector.py` | 6 tests | English, Hindi, Tamil detection, fallback on failure |
| `test_language/test_translator.py` | 4 tests | Hindi→English, Tamil→English, fallback model, batch |
| `test_search/test_hybrid.py` | 8 tests | RRF fusion, parallel search, hybrid vs vector-only vs keyword-only |
| `test_search/test_vector.py` | 5 tests | Build index, search, save/load, empty index, dimension mismatch |
| `test_search/test_bm25.py` | 5 tests | Build index, search, save/load, tokenization |
| `test_search/test_reranker.py` | 4 tests | Rerank improves precision, timeout fallback, score ordering |
| `test_matching/test_skill_matcher.py` | 8 tests | Exact, fuzzy, semantic matching, proficiency scoring, required vs optional |
| `test_matching/test_scorer.py` | 5 tests | Overall score, weight normalization, null handling, confidence |
| `test_agents/test_orchestrator.py` | 5 tests | Full pipeline, re-plan, max cycles, LLM fallback |
| `test_agents/test_planner.py` | 4 tests | Parse query, JSON output, fallback, replan |
| `test_agents/test_reflector.py` | 4 tests | Evaluate matches, should_replan, feedback generation |
| `test_rationale/test_generator.py` | 4 tests | Generate rationale, batch, fallback, validation |
| `test_api/test_search_endpoint.py` | 6 tests | POST /search, validation, empty results, error handling |
| `test_api/test_health_endpoint.py` | 3 tests | Health check, index size, model status |
| `test_integration/test_end_to_end.py` | 6 tests | Full pipeline, multilingual, messy data, latency, no-results, error handling |

---

## Phase 15: Documentation

### 15.1 `README.md`

```markdown
# 🏃 India Runs — Intelligent Candidate Discovery

> Hackathon submission for India Runs by Redrob AI — Track 1: Data & AI Challenge

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- OpenAI API key (for agentic workflow)

### Setup
\```bash
# 1. Clone and install
git clone <repo-url>
cd india-runs
pip install -e ".[dev]"

# 2. Start infrastructure
docker compose up -d postgres redis

# 3. Set up environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 4. Download spacy model
python -m spacy download en_core_web_sm

# 5. Generate synthetic data
python scripts/generate_data.py

# 6. Build indexes
python scripts/build_indexes.py

# 7. Run evaluation
python scripts/evaluate.py

# 8. Start the application
uvicorn src.main:app --reload
# OR
python src/ui/app.py
\```

## Architecture

See [docs/architecture.md](docs/architecture.md)

## API Documentation

Once running, visit: http://localhost:8000/docs

## Running Tests

\```bash
pytest tests/ -v --cov=src
\```

## Tech Stack

- **Search**: FAISS (vector) + BM25 (keyword) + RRF fusion + Cross-Encoder reranking
- **Agents**: LangGraph (Plan → Execute → Reflect → Re-plan)
- **ML**: sentence-transformers, torch, langdetect, spacy
- **API**: FastAPI + Uvicorn
- **UI**: Gradio 5.0
- **Data**: PostgreSQL + Redis
```

### 15.2 `docs/architecture.md`

~2000 word architecture deep-dive explaining:
1. System overview diagram
2. Data flow (ingestion → indexing → search → scoring → rationale)
3. Agentic workflow state machine
4. Hybrid search pipeline
5. Multilingual processing
6. Fairness considerations
7. Performance optimizations

### 15.3 `docs/api.md`

API documentation with curl examples for all 4 endpoints.

### 15.4 `docs/evaluation.md`

How to run evaluation, interpret metrics, and understand the evaluation notebook.

### 15.5 `docs/deployment.md`

Step-by-step deployment guide for:
1. Local development
2. Docker deployment
3. Gradio Live (HuggingFace Spaces)
4. Railway/Render

---

## Execution Order

The implementation must follow this exact dependency order:

```
Phase 0  (Environment)     ← No dependencies
    ↓
Phase 1  (Core/Config)     ← Depends on: Phase 0
    ↓
Phase 2  (Data Gen)        ← Depends on: Phase 1
    ↓
Phase 3  (Ingestion)       ← Depends on: Phase 1
    ↓
Phase 4  (Language)        ← Depends on: Phase 1
    ↓
Phase 5  (Search)          ← Depends on: Phase 4 (embeddings)
    ↓
Phase 6  (Matching)        ← Depends on: Phase 1
    ↓
Phase 7  (Agents)          ← Depends on: Phase 5, 6
    ↓
Phase 8  (Rationale)       ← Depends on: Phase 7
    ↓
Phase 9  (Fairness)        ← Depends on: Phase 6
    ↓
Phase 10 (API)             ← Depends on: Phase 7, 8
    ↓
Phase 11 (UI)              ← Depends on: Phase 10
    ↓
Phase 12 (Index Build)     ← Depends on: Phase 2, 4, 5
    ↓
Phase 13 (Evaluation)      ← Depends on: Phase 12
    ↓
Phase 14 (Testing)         ← After all phases
    ↓
Phase 15 (Documentation)   ← After all phases
```

**Within each phase, create files in the order listed.**

---

## File Count Summary

| Phase | Files | Lines (est.) |
|-------|-------|-------------|
| 0: Environment | 6 | ~150 |
| 1: Core | 5 | ~350 |
| 2: Data | 4 | ~500 |
| 3: Ingestion | 4 | ~400 |
| 4: Language | 3 | ~300 |
| 5: Search | 5 | ~500 |
| 6: Matching | 4 | ~400 |
| 7: Agents | 5 | ~600 |
| 8: Rationale | 3 | ~300 |
| 9: Fairness | 2 | ~250 |
| 10: API | 7 | ~400 |
| 11: UI | 3 | ~500 |
| 12: Index Build | 1 | ~80 |
| 13: Evaluation | 1 | ~200 |
| 14: Testing | 19 | ~1500 |
| 15: Documentation | 6 | ~800 |
| **Total** | **78** | **~7,230** |

---

*Every file, every function, every test. No shortcuts. No MVPs.*
*This is the finished product — ready to build, ready to deploy.*
