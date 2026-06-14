# Architecture — India Runs

## System Overview

India Runs is a modular candidate discovery system built for the Redrob AI hackathon. The architecture follows a pipeline pattern with 5 major stages: **Ingestion → Indexing → Search → Matching → Presentation**, plus an **Agentic Workflow** that orchestrates iterative search refinement.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Ingestion  │ →  │  Indexing   │ →  │   Search    │ →  │  Matching   │
│ (JSONL/JSON)│    │ (FAISS+BM25)│    │ (Hybrid+RRF)│    │ (Scorer)    │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  Gradio UI  │ ←  │  FastAPI    │ ←  │  Rationale  │ ←────────┘
│  (Frontend) │    │  (REST)     │    │  (Explain)  │
└─────────────┘    └─────────────┘    └─────────────┘
       ↑                  ↑
       └────── Users ─────┘
```

## Data Flow

### 1. Ingestion (`src/ingestion/`)
- **Parser**: Streams JSONL/JSON files (supports gzip). 100K profiles processed without loading into memory.
- **Normalizer**: Maps Redrob API schema (8 top-level fields) to the internal Profile model (30+ fields). Handles nested profile objects, skills with proficiency, career history, education, certifications, and signals.
- **Quality Scorer**: Computes a 0-1 data quality score based on field completeness (name, title, company, skills, experience, education).

### 2. Indexing (`scripts/build_indexes.py`)
- **FAISS Index**: Dense vector search using Inner Product on normalized 384-dim embeddings from paraphrase-multilingual-MiniLM-L12-v2.
- **BM25 Index**: Sparse keyword search using rank_bm25 with case-insensitive tokenization.
- Both indexes are built once and saved to disk. The FAISS index includes a JSON id_map for profile-to-vector lookups.

### 3. Search Pipeline (`src/search/`)
- **Hybrid Search**: Runs FAISS and BM25 searches in parallel, then fuses results using Reciprocal Rank Fusion (RRF) with configurable k parameter.
- **Cross-Encoder Reranker**: Reranks top-k hybrid results using ms-marco-MiniLM-L-6-v2 cross-encoder. Configurable timeout.
- **Filters**: Applies hard filters (location, experience range, company inclusion/exclusion) before ranking.

### 4. Matching & Scoring (`src/matching/`)
- **Skill Matcher**: 4-strategy matching pipeline (exact → normalized → alias → fuzzy). 32 curated skill aliases (e.g., "react" ↔ "react.js"). Proficiency level scoring.
- **Experience Matcher**: Years-of-experience scoring with configurable deficit/excess penalty. Industry match scoring (exact 1.0, else 0.3).
- **Scorer**: Weighted combination of 6 dimensions (semantic similarity, keyword match, skill match, experience match, confidence, normalization). Weights loaded from configs/scoring_weights.yaml at runtime.

### 5. Agentic Workflow (`src/agents/`)

The system uses a LangGraph state machine with 4 stages:

```
Plan → Execute → Reflect → Re-plan (conditional loop)
```

- **Planner**: Receives a job query, extracts structured search parameters using LLM (or fallback keyword extraction). Generates a plan with skills, title, experience, location, and companies. Supports re-planning with relaxed criteria.
- **Executor**: Runs the full search pipeline (hybrid search + filters + reranker + scorer). Returns ranked MatchResult list.
- **Reflector**: Evaluates results using LLM or score-threshold heuristics. Determines whether to accept results or re-plan with broader criteria.
- **Orchestrator**: State machine managing the Plan→Execute→Reflect→Re-plan loop. Configurable max_replan_cycles. Terminal states: SUCCESS, GIVE_UP.

### 6. Rationale Generation (`src/rationale/`)
- **Generator**: Uses LLM (provider-agnostic) or template fallback to produce human-readable explanations for each match.
- **Validator**: Checks rationale quality (summary length, strengths, valid recommendation).
- Output includes: summary, strengths, gaps, skill-level detail, experience analysis, and recommendation (Strong/Good/Weak/Poor).

### 7. Fairness & Bias (`src/fairness/`)
- **Bias Detector**: Analyzes 4 bias dimensions:
  - Name bias: first-character grouping of candidate names
  - Language bias: score comparison between English and non-English profiles
  - Location bias: tier-1 vs tier-2/3 city candidates
  - University bias: IIT/NIT/BITS vs other institutions
- **Metrics**: Computes demographic parity, disparate impact ratio (4/5ths rule), and aggregate fairness dashboard.

## Multilingual Processing

The language pipeline supports 12 Indian languages. In practice, all 100K Redrob profiles are English, so translation is a no-op for this dataset. The architecture correctly handles the case through lazy model loading and fallback chains.

- **Language Detector**: langdetect for identification
- **Translation Pipeline**: Stub implementation — real opus-mt models are ~300MB each per language pair; loading all 9 would exceed the 16GB/5min constraint
- **Multilingual Embedder**: paraphrase-multilingual-MiniLM-L12-v2 (384-dim, 50+ languages)

## Performance Optimizations

- **CPU-optimized**: All models run on CPU (no GPU requirement). sentence-transformers with ONNX runtime and 256-token truncation.
- **Streaming**: JSONL parser streams line-by-line — never loads entire dataset into memory.
- **Caching**: Settings, configs, and models use @lru_cache for singleton access.
- **Parallelism**: FAISS and BM25 searches run in parallel during hybrid search.
- **Timeout safety**: Cross-encoder reranker has configurable timeout (default 500ms) to prevent pipeline stalls.

## Configuration

All configuration is externalized to YAML files with env var interpolation:
- `configs/settings.yaml`: App settings, database, search parameters, agent limits
- `configs/scoring_weights.yaml`: Scoring weights for 6 dimensions, proficiency scores, skill importance
- `configs/models.yaml`: Model names, dimensions, devices for embeddings, cross-encoder, translation

## LLM Provider Abstraction

The system supports 3 LLM providers through a unified factory:
- **OpenAI** (default, gpt-4o-mini)
- **Google Gemini** (gemini-2.0-flash via langchain-google-genai)
- **Local Ollama** (llama3.1:8b via langchain-ollama)

Provider is configured via the `LLM_PROVIDER` env var. The factory in `src/core/config.py` handles all provider-specific initialization.
