# India Runs — Intelligent Candidate Discovery

> **Track 1: The Data & AI Challenge** — India Runs by Redrob AI × Hack2Skill
>
> Multi-query candidate ranking pipeline with hybrid semantic search, cross-encoder reranking, and 7-dimensional weighted scoring. Runs fully offline on CPU in under 8 seconds.

## Quick Start

### Prerequisites
- Python 3.11+
- 16GB RAM (CPU only — no GPU required)
- No network required after model caching

### Setup

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Build indexes (100 sample profiles)
python scripts/build_indexes.py

# 3. Generate submission CSV
python rank.py --out submission.csv

# 4. Validate output
python validate_submission.py submission.csv
```

**Reproduction command** (per submission spec):
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

## How It Works

### Pipeline Overview

```
17 Strategic Queries → Hybrid Search (FAISS + BM25) → Cross-Encoder Reranking → 7-Dimension Scoring → Deduplicated Merge → Ranked CSV
```

### Multi-Query Strategy

The system runs **17 diverse queries** covering distinct roles and tech stacks:

| Category | Queries |
|----------|---------|
| Core Engineering | software engineer (Python/Java/React), backend (Python/Django/AWS/SQL), frontend (React/TypeScript), full stack (React/Node.js/MongoDB) |
| Data & ML | data scientist (Python/PyTorch/SQL), data engineer (Spark/Airflow/Kafka), ML engineer (deep learning/NLP/CV) |
| Cloud & DevOps | devops engineer (Docker/K8s/AWS/Terraform), cloud architect (AWS/Azure/GCP) |
| Mobile & Backend | mobile (Android/Kotlin/Swift/React Native), Java (Spring Boot/Microservices), Python (FastAPI/Flask/Django) |
| Leadership | engineering manager (distributed systems), product manager (analytics/roadmap) |
| Domain | cybersecurity, QA automation, solutions architect |

Each query retrieves up to 50 candidates. Results are merged with deduplication, keeping the **best score per candidate** across all queries.

### Scoring (7 Dimensions)

| Dimension | Weight | Source |
|-----------|--------|--------|
| Skill Match | 25% | Fuzzy alias matching + semantic overlap |
| Cross-Encoder Score | 25% | ms-marco-MiniLM-L-6-v2 relevance |
| Semantic Similarity | 20% | FAISS cosine distance (384-dim) |
| Keyword Match | 10% | BM25 Okapi score |
| Experience Match | 10% | Years of experience vs requirement |
| Location Match | 5% | City-based matching |
| Education Match | 5% | Degree + institution quality |

Weights are configurable in `configs/scoring_weights.yaml`.

### Key Components

1. **Hybrid Search** — FAISS (Inner Product, 384-dim) + BM25 Okapi with Reciprocal Rank Fusion (k=60)
2. **Cross-Encoder Reranker** — `cross-encoder/ms-marco-MiniLM-L-6-v2` for fine-grained query-document relevance. Pre-loaded eagerly at startup; runs in offline mode.
3. **Multi-Dimension Scorer** — Weighted combination of 7 signals (skill overlap, semantic similarity, keyword match, cross-encoder relevance, experience, location, education)
4. **Honeypot Resistance** — Cross-encoder's deep profile-text comprehension naturally filters out candidates with internally inconsistent profiles (e.g., 8 years experience at a 3-year-old company)

### Performance

- **Full pipeline**: ~7.5 seconds (17 queries × 50 candidates each)
- **Cross-encoder load time**: ~0.3s (cached locally)
- **Memory**: Under 4GB RAM
- **Network**: Zero — all models cached and run offline

## Submission Files

| File | Description |
|------|-------------|
| `rank.py` | Main entry point (`python rank.py --candidates ./candidates.jsonl --out ./submission.csv`) |
| `submission.csv` | 100 ranked candidates (validated) |
| `validate_submission.py` | Hackathon format validator |
| `submission_metadata.yaml` | Submission metadata for portal |
| `src/search/reranker.py` | Cross-encoder reranker (enabled, pre-loaded) |
| `src/agents/executor.py` | Executor agent with score-sorted ranking |
| `src/agents/orchestrator.py` | Query parsing with city-skill separation |

## Architecture

```
src/
├── core/                   # Pydantic models, config, constants
├── ingestion/             # Profile parsing & normalization
├── language/              # Multilingual embeddings
├── search/                # FAISS, BM25, hybrid search, cross-encoder reranker
├── matching/              # Skill alias matching, 7-dimension scorer
├── agents/                # Planner, Executor, Orchestrator (LangGraph)
├── ranking/               # Plackett-Luce listwise ranker
├── rationale/             # Template-based rationale generator
├── fairness/              # Bias detection & PII anonymization
├── api/                   # FastAPI REST endpoints
└── ui/                    # Gradio interactive dashboard
```

## Running Tests

```bash
python -m pytest tests/ -q --tb=short
# → 150 passed
```

## AI Tools Declaration

Used Claude for architectural design, code review, and test-driven development.
GitHub Copilot for autocomplete during implementation.
No candidate data was fed to any external LLM after dataset release.

## License

This project is submitted to the India Runs by Redrob AI hackathon (Track 1).
