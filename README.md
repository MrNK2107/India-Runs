# India Runs — Intelligent Candidate Discovery

> 🏆 **Track 1: Data & AI Challenge** — India Runs by Redrob AI
>
> Intelligent Candidate Discovery & Ranking System with agentic AI matching, 100+ profiles, explainable rationale, and submission-ready output.

## Key Differentiators

| Feature | What it does |
|---|---|
| **Hybrid Semantic Search** | FAISS vector (384-dim) + BM25 Okapi + RRF fusion — finds candidates by meaning AND keywords |
| **Plackett-Luce Listwise Ranking** | Candidates compete in tournament groups judged by an LLM, aggregated via MM algorithm |
| **Agentic Workflow (LangGraph)** | Plan → Execute → Listwise Rank → Reflect → Re-plan (up to 3 cycles) with fallback |
| **Multilingual Indian Queries** | 30+ Indian languages supported. Code-mixed Hinglish queries handled via TinT prompting |
| **Interactive Scoring Sliders** | 6 recruiter-facing dimensions — adjust weights, results re-rank instantly |
| **Every Match Explained** | Template-based rationale with skill match table, strengths, gaps, recommendation |
| **PII Anonymization** | Names, emails redacted before LLM evaluation — prevents bias |
| **Live Fairness Dashboard** | Real-time bias metrics (university, city, language parity) in Gradio |
| **Ollama Local LLM** | Runs fully offline with qwen3:4b — no API keys needed |

## Quick Start

### Prerequisites
- Python 3.11+
- Ollama (optional — for LLM-powered features; fallback parser works without it)
- NVIDIA GPU with 6GB+ VRAM (optional — CPU mode works, GPU accelerates embeddings)

### Setup

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Build indexes (100 sample profiles)
python scripts/build_indexes.py

# 3. Launch FastAPI server (port 8000)
uvicorn src.main:app --host 0.0.0.0 --port 8000

# 4. Launch Gradio UI (port 7860) — OR use FastAPI
python src/ui/app.py
```

### Generate Submission CSV

```bash
# Generates submission.csv with 100 ranked candidates
python generate_submission.py

# Validate output
python validate_submission.py submission.csv
```

## Quick Test

```bash
# Search via API (simple query — 0.5s response)
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "software engineer python java", "max_results": 5}'

# Full agentic pipeline query (~60s with Ollama)
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "I need a senior backend developer with Python and AWS in Bangalore"}'

# Health check
curl http://localhost:8000/api/v1/health
```

## Architecture

```
Profiles (JSON) → Normalizer → Embedder → FAISS + BM25 Indexes
                                      ↓
Job Query → Planner → Hybrid Search → Reranker → Scorer → Plackett-Luce Ranker
                          ↓                                          ↓
                     Reflector ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← 
                          ↓
                     Rationale Generator → Final Ranked Output → submission.csv
```

### Pipeline Stages

1. **Planner** — LLM (Ollama qwen3:4b) or rule-based fallback parses query into structured params
2. **Executor** — Hybrid search (FAISS + BM25 + RRF) → Optional cross-encoder → 7-dimension scoring
3. **Plackett-Luce Ranker** — Tournament-based listwise ranking with LLM judges (disabled by default for speed)
4. **Reflector** — LLM evaluates result quality; triggers re-plan if < threshold good matches
5. **Rationale Generator** — Template-based explanation for every candidate (always works)

### Scoring Dimensions

| Dimension | Source | Configurable |
|---|---|---|
| Skill Match | Fuzzy alias matching + semantic overlap | ✔ via slider |
| Experience Match | Years of experience vs requirement | ✔ via slider |
| Education Match | Degree level + institution tier | ✔ via slider |
| Semantic Similarity | FAISS cosine distance | — |
| Keyword Match | BM25 Okapi score | — |
| Cross-Encoder Score | MiniLM-L12 cross-encoder (disabled by default) | — |
| Assessment Score | Synthetic (for recruiter input) | ✔ via slider |
| Behavioral Signals | Profile-derived signals | ✔ via slider |
| Cultural Fit | Industry/domain alignment | ✔ via slider |

### Tech Stack
- **Search**: FAISS (IVF, 384-dim) + BM25 Okapi + RRF
- **Ranking**: Plackett-Luce listwise model with MM aggregation
- **Agents**: LangGraph state machine (Plan → Execute → Reflect → Re-plan)
- **ML**: sentence-transformers, FAISS, rank_bm25, scikit-learn
- **LLM**: Ollama (qwen3:4b) — runs locally, no API keys
- **API**: FastAPI + Uvicorn (port 8000)
- **UI**: Gradio (port 7860) — search, sliders, rationale, live analytics

## Running Tests

```bash
pytest tests/ -v
# → 150 tests passing
```

## Project Structure

```
india-runs/
├── configs/                    # YAML configuration files
│   ├── settings.yaml           # Application settings
│   ├── scoring_weights.yaml    # Scoring dimension weights
│   └── models.yaml             # ML model configurations
├── src/
│   ├── core/                   # Pydantic models, config, constants
│   │   ├── models.py          # 33 Pydantic models (Profile, MatchResult, etc.)
│   │   ├── config.py          # Settings, LLM client factory
│   │   └── constants.py       # 200+ Indian companies, cities, universities
│   ├── ingestion/             # Profile parsing, normalization
│   ├── language/              # Code-mixed detection, multilingual embeddings
│   ├── search/                # FAISS, BM25, hybrid search, reranker
│   ├── matching/              # Skill alias matching, weighted scorer
│   ├── agents/                # LangGraph: Planner, Executor, Reflector, Orchestrator
│   ├── ranking/               # Plackett-Luce listwise tournament ranker
│   ├── rationale/             # Template-based rationale generator
│   ├── fairness/              # Bias detector, metrics, PII anonymizer
│   ├── api/                   # FastAPI routes (search, health, profiles)
│   └── ui/                    # Gradio app with live analytics dashboard
├── scripts/                   # Index builder, evaluation harness
├── tests/                     # 150 unit/integration tests
├── data/
│   ├── samples/               # 100 sample profiles for demo
│   └── indexes/               # FAISS + BM25 indexes
├── submission.csv             # Generated submission (100 ranked candidates)
├── generate_submission.py     # Submission generator script
└── validate_submission.py     # Hackathon output validator
```
