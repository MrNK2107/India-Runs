# India Runs — Intelligent Candidate Discovery

> Hackathon submission for India Runs by Redrob AI — Track 1: Data & AI Challenge

Intelligent Candidate Discovery & Ranking System processing 100K real Redrob profiles. Hybrid semantic search with agentic AI matching, fairness-aware scoring, and explainable rationale generation.

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- OpenAI API key (for agentic workflow, optional)

### Setup

```bash
# 1. Clone and install
pip install -e ".[dev]"

# 2. Start infrastructure
docker compose up -d postgres redis

# 3. Set up environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 4. Build indexes
python scripts/build_indexes.py

# 5. Run evaluation
python scripts/evaluate.py

# 6. Start the API
uvicorn src.main:app --reload

# 7. Or start the Gradio UI
python src/ui/app.py
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for a deep dive.

### Pipeline Overview

```
Profiles (JSONL) → Normalizer → Embedder → FAISS + BM25 Indexes
                                      ↓
Job Query → Planner → Hybrid Search → Reranker → Scorer → Results
                          ↓                                   ↓
                     Reflector                         Rationale Generator
                          ↓                                   ↓
                     Re-plan (loop)                   Final Ranked Output
```

### Tech Stack
- **Search**: FAISS (vector) + BM25 (keyword) + RRF fusion + Cross-Encoder reranking
- **Agents**: LangGraph state machine (Plan → Execute → Reflect → Re-plan)
- **ML**: sentence-transformers, langdetect, FAISS, rank_bm25
- **API**: FastAPI + Uvicorn
- **UI**: Gradio

## API Documentation

Once running, visit [http://localhost:8000/docs](http://localhost:8000/docs) or see [docs/api.md](docs/api.md).

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## Project Structure

```
india-runs/
├── configs/              # YAML configuration files
│   ├── settings.yaml     # Application settings
│   ├── scoring_weights.yaml  # Scoring dimension weights
│   └── models.yaml       # ML model configurations
├── src/
│   ├── core/             # Models, config, constants
│   ├── ingestion/        # Profile parsing, normalization
│   ├── language/         # Language detection, translation, embeddings
│   ├── search/           # FAISS, BM25, hybrid, reranker, filters
│   ├── matching/         # Skill/experience matching, scoring
│   ├── agents/           # LangGraph agentic workflow
│   ├── rationale/        # Explainable rationale generation
│   ├── fairness/         # Bias detection & fairness metrics
│   ├── api/              # FastAPI endpoints & middleware
│   └── ui/               # Gradio application
├── scripts/              # Index building, evaluation
├── tests/                # 57 unit/integration tests
└── docs/                 # Architecture, API, deployment docs
```
