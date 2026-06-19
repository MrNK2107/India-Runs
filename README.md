# India Runs — Intelligent Candidate Discovery

> 🏆 Hackathon submission for **India Runs by Redrob AI** — Track 1: Data & AI Challenge

Intelligent Candidate Discovery & Ranking System. Hybrid semantic search over 50 sample profiles (scales to 100K+) with **agentic AI matching**, **Plackett-Luce listwise tournament ranking**, **PII anonymization**, **live fairness analytics**, and **explainable rationale** for every match.

## Key Differentiators

| Feature | What it does |
|---|---|
| **Plackett-Luce Listwise Ranking** | Candidates compete in tournament groups judged by an LLM, aggregated via Minorization-Maximization (MM) algorithm — produces global merit scores from partial rankings |
| **PII Anonymization** | Names, emails, phone numbers stripped before LLM evaluation — prevents name/IIT/NIT-based bias. Style normalization removes LLM-sounding verbs ("leveraged" → "used") |
| **Live Fairness Dashboard** | Real-time bias metrics (university parity, city parity, language parity, rank distribution) computed from search results, displayed in Gradio analytics tab |
| **Agentic Workflow (LangGraph)** | Plan → Execute → Listwise Rank → Reflect → Re-plan (up to 3 cycles) with fallback to rule-based parsing when LLM unavailable |
| **Hinglish Code-Mixed Queries** | TinT (Translate-in-Thought) prompting for Hindi-English mixed queries. Multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2) covering 30+ Indian languages |
| **Interactive Scoring Sliders** | 6 recruiter-facing dimensions (Skill, Experience, Education, Assessment, Behavioral, Cultural Fit) — adjust weights, results re-rank instantly without re-searching |
| **Every Match Explained** | Template-based rationale (always works) with skill match table, strengths, gaps, experience analysis, and recommendation |
| **Bias Detection** | Demographic parity, disparate impact ratio, language bias, location bias, university bias — all computed from actual results |

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (optional — fallback parser works without it)

### Setup

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Build indexes with sample data (50 profiles)
python scripts/build_indexes.py --sample 50

# 3. Launch Gradio UI
python src/ui/app.py
# → Opens at http://localhost:7860
```

## Architecture

```
Profiles (JSON) → Normalizer → Embedder → FAISS + BM25 Indexes
                                      ↓
Job Query → Planner → Hybrid Search → Reranker → Scorer → Plackett-Luce Ranker
                          ↓                                          ↓
                     Reflector ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← 
                          ↓
                     Rationale Generator → Final Ranked Output
```

### Pipeline Stages

1. **Planner** — LLM parses query into structured params (skills, experience, location); falls back to regex-based parsing
2. **Executor** — Hybrid search (FAISS vector + BM25 keyword + RRF fusion) → Cross-encoder reranking → Multidimensional scoring
3. **Plackett-Luce Ranker** — Candidates grouped into tournaments, LLM judges groups listwise, MM algorithm aggregates into global merit scores
4. **Reflector** — LLM evaluates result quality; triggers re-plan with relaxed params if too few good matches
5. **Rationale Generator** — Template-based explanation for every candidate (always works, no API key needed)

### Scoring Dimensions

| Dimension | Source | Configurable |
|---|---|---|
| Skill Match | Fuzzy alias matching + semantic overlap | ✔ via slider |
| Experience Match | Years of experience vs requirement | ✔ via slider |
| Education Match | Degree level + institution tier | ✔ via slider |
| Semantic Similarity | FAISS cosine distance | — |
| Keyword Match | BM25 Okapi score | — |
| Cross-Encoder Score | MiniLM-L12 cross-encoder | — |
| Assessment Score | Synthetic (for recruiter input) | ✔ via slider |
| Behavioral Signals | Profile-derived signals | ✔ via slider |
| Cultural Fit | Industry/domain alignment | ✔ via slider |

### Tech Stack
- **Search**: FAISS (IVF, 384-dim) + BM25 Okapi + RRF + Cross-Encoder reranking
- **Ranking**: Plackett-Luce listwise model with MM aggregation
- **Agents**: LangGraph state machine (Plan → Execute → Listwise Rank → Reflect → Re-plan)
- **ML**: sentence-transformers, langdetect, FAISS, rank_bm25, scikit-learn
- **Fairness**: Demographic parity, disparate impact, bias detection (name/language/location/university)
- **API**: FastAPI + Uvicorn (port 8000)
- **UI**: Gradio (port 7860) — search, sliders, rationale, live analytics dashboard

## Fairness & Transparency

- **PII Anonymization Pipeline**: Every profile is anonymized before LLM evaluation — name replaced with hash, email/phone redacted, education tier and location tier used instead of raw institution/city names
- **Bias Monitoring**: 4 real-time metrics track demographic parity across university, city, and language dimensions
- **Explainable Rankings**: Every result includes skill-level rationale with strengths, gaps, evidence, and recommendation

## Running Tests

```bash
pytest tests/ -v
# → 149 tests passing (100% without API key)
```

## Project Structure

```
india-runs/
├── configs/                    # YAML configuration files
│   ├── settings.yaml           # Application settings
│   ├── scoring_weights.yaml    # Scoring dimension weights + PL config
│   └── models.yaml             # ML model configurations
├── src/
│   ├── core/                   # Pydantic models, config, constants
│   │   ├── models.py          # 33 Pydantic models (Profile, MatchResult, etc.)
│   │   ├── config.py          # Settings, LLM client factory
│   │   └── constants.py       # 200+ Indian companies, cities, universities
│   ├── ingestion/             # Profile parsing, normalization, quality scoring
│   ├── language/              # Code-mixed detection, translation, multilingual embeddings
│   ├── search/                # FAISS, BM25, hybrid search, reranker, filters
│   ├── matching/              # Skill alias matching, experience matching, weighted scorer
│   ├── agents/                # LangGraph: Planner, Executor, Reflector, Orchestrator
│   ├── ranking/               # Plackett-Luce listwise tournament ranker ✨
│   ├── rationale/             # Template-based + LLM rationale generator
│   ├── fairness/              # Bias detector, metrics, PII anonymizer ✨
│   ├── api/                   # FastAPI routes (search, health, profiles, ingest)
│   └── ui/                    # Gradio app with live analytics dashboard ✨
├── scripts/                   # Index builder, evaluation harness
├── tests/                     # 149 unit/integration tests
└── data/
    ├── samples/               # 50 sample profiles for demo
    └── indexes/               # FAISS + BM25 indexes
