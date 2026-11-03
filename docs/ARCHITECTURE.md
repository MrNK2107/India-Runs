# Kasauti — architecture and hackathon notes

The deep dive behind [the README](../README.md): the retrieval and scoring pipeline, the
agentic (LLM) path, the CLI used for the India Runs hackathon submission, configuration,
and the original benchmark numbers. Kasauti began as the *India Runs* Track 1 entry by team
PANGDynamics; the explainability layer, web app and lite engine are documented in
[PRODUCT.md](PRODUCT.md).

## Explainability layer

`src/explain/evidence.py` runs after scoring and adds an `Explanation` to every search result.
It has no model and is deterministic:

- **contributions**: per dimension, `weight / Σweights × score × title_factor`. These sum to
  `overall`. `title_factor` is the executor's role-fit multiplier, recovered as
  `overall / weighted_mean`.
- **requirements**: query skills (required, then preferred), the role, the experience band and
  the city, each with `status` (met / partial / missing), `strength`
  (verified > structured > demonstrated > mentioned) and `evidence` spans `{source, label,
  text, start, end}`. `text[start:end]` is the matched term. Skill aliases come from
  `SKILL_ALIASES`; aliases shorter than 3 characters are not searched in prose.
- **strengths / risks**: from platform signals, honeypot checks, tenure, and the claim-vs-proof gap.

`POST /api/v1/explain` scores and explains one profile for a query even if it is not in the
result list ("why isn't X ranked?").

## Lite engine

`EMBEDDING_BACKEND=hashing` swaps the sentence-transformer for `src/language/hashing.py`:
signed feature hashing of word unigrams (1.0), bigrams (0.7) and character trigrams (0.25)
into 384 dimensions, log-scaled and L2-normalised, with a small synonym fold (bengaluru →
bangalore, k8s → kubernetes, …). With no FAISS index on disk, `src/core/engine.py` indexes
`data/samples/sample_500.jsonl` in memory at startup (about 4 s). torch is imported lazily, so
the lite runtime never loads it (`requirements-lite.txt`).

## 🏗️ System Architecture

### Core Pipeline

```
                     ┌─────────────────────────────────────────────┐
                     │             USER QUERY                      │
                     │  (natural language or structured query)      │
                     └──────────────────┬──────────────────────────┘
                                        ▼
                     ┌─────────────────────────────────────────────┐
                     │         PLANNER AGENT (LangGraph)           │
                     │  • LLM-based intent parsing                 │
                     │  • TinT prompting for Hinglish queries      │
                     │  • Rule-based fallback for simple queries   │
                     └──────────────────┬──────────────────────────┘
                                        ▼
                     ┌─────────────────────────────────────────────┐
                     │         EXECUTOR AGENT                      │
                     │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
                     │  │  FAISS   │  │   BM25   │  │  Hybrid  │  │
                     │  │  HNSW    │  │  Keyword │  │  RRF     │  │
                     │  │  Vector  │  │  Search  │  │  Fusion  │  │
                     │  └──────────┘  └──────────┘  └──────────┘  │
                     │              │                              │
                     │              ▼                              │
                     │  ┌────────────────────────────────────┐     │
                     │  │    CROSS-ENCODER RERANKER          │     │
                     │  │  ms-marco-MiniLM-L-6-v2            │     │
                     │  │  Deep bidirectional attention       │     │
                     │  └────────────────────────────────────┘     │
                     │              │                              │
                     │              ▼                              │
                     │  ┌────────────────────────────────────┐     │
                     │  │    10-DIMENSION SCORING ENGINE     │     │
                     │  │  Weighted fusion with honeypot     │     │
                     │  │  penalties and confidence scoring  │     │
                     │  └────────────────────────────────────┘     │
                     └──────────────────┬──────────────────────────┘
                                        ▼
                     ┌─────────────────────────────────────────────┐
                     │     LISTWISE RANKER (Plackett-Luce)         │
                     │  Pairwise tournament with LLM group judge   │
                     └──────────────────┬──────────────────────────┘
                                        ▼
                     ┌─────────────────────────────────────────────┐
                     │         REFLECTOR AGENT                     │
                     │  • Quality evaluation • Re-plan decision    │
                     └──────────────────┬──────────────────────────┘
                                        ▼
                     ┌─────────────────────────────────────────────┐
                     │     RATIONALE GENERATOR                     │
                     │  Signal-driven, non-templated per candidate │
                     └──────────────────┬──────────────────────────┘
                                        ▼
                     ┌─────────────────────────────────────────────┐
                     │           RANKED RESULTS                    │
                     │  Score + rank + detailed reasoning           │
                     └─────────────────────────────────────────────┘
```

### Scoring Dimensions

| # | Dimension | Weight | What It Measures |
|---|---|---|---|---|
| 1 | **Cross-encoder score** | 15% | Deep semantic match via bidirectional transformer attention |
| 2 | **Skill match** | 20% | Fuzzy skill matching with 32+ alias groups and synonym resolution |
| 3 | **Semantic similarity** | 20% | FAISS cosine similarity on multilingual embeddings (384-dim) |
| 4 | **Behavioral signals** | 12% | Recruiter response rate, saves, completeness, verification, GitHub activity, interview conversion |
| 5 | **Keyword match** | 10% | BM25 term overlap on raw profile text |
| 6 | **Experience match** | 8% | Total experience vs target band with deficit/excess penalties |
| 7 | **Career trajectory** | 7% | Job hopping (<18mo avg × 3+ jobs), consulting penalty, title progression |
| 8 | **Skill proficiency** | 5% | Depth-weighted: expert > advanced > intermediate > beginner |
| 9 | **Location match** | 0%* | Reserved for recruiter slider |
| 10 | **Education match** | 0%* | Reserved for recruiter slider |

*\* Configurable in UI via interactive sliders*

---

## 🧠 Intelligent Features

### 🔍 Hybrid Semantic Search
Combines the best of both worlds:
- **FAISS HNSW** (Hierarchical Navigable Small World) — fast approximate nearest neighbor search on 384-dim multilingual embeddings
- **BM25 Okapi** — proven keyword retrieval with `np.argpartition` top-k optimization
- **Reciprocal Rank Fusion (RRF)** — optimal fusion of vector + keyword results

### 🔄 Agentic Workflow (LangGraph)
```
Plan ──► Execute ──► Listwise Rank ──► Reflect ──► Generate
  │                                              │
  └────────── Re-plan (max 3 cycles) ────────────┘
```
- **Planner Agent**: LLM-based intent extraction with TinT prompting for Hinglish
- **Executor Agent**: Parallel profile loading (16 workers), batch cross-encoder scoring
- **Reflector Agent**: Quality evaluation with automated re-plan decision
- **Listwise Ranker**: Plackett-Luce tournament ranking with LLM group judge

### 🚨 Honeypot Detection
Protects ranking integrity by identifying impossible/fake profiles:
1. **Time-travel**: Start year before company founded (e.g., "worked at Pied Piper in 2012")
2. **Skill-density anomaly**: >5 skills per year of total experience
3. **Expert-zero-years**: Expert proficiency in 5+ skills with 0 years used
4. **Long unexplained gap**: more than 5 years between consecutive jobs

Penalty: **×0.15 score multiplier** — naturally sinks to bottom ranks.

### 🌐 Multilingual & Hinglish Support
- **50+ languages** via `paraphrase-multilingual-MiniLM-L12-v2`
- **Hinglish detection** with code-mixed processor (Devanagari + Latin script analysis)
- **TinT (Translate-in-Thought) prompting** — LLM internally translates Hinglish before parsing
- **Google Translate + MBart-50 fallback** for non-English content
- **Language-agnostic embeddings** — search across languages seamlessly

### 🛡️ Fairness & Bias Mitigation
- **PII Anonymization**: Automatic redaction of names, emails, phone numbers before ranking
- **Style Anonymization**: Removes identifying writing patterns (LLM-based)
- **Bias Detection**: Real-time monitoring across university, location, and language dimensions
- **Disparate Impact Ratio (DIR)**: Statistical parity measurement (threshold: <0.80 triggers alert)
- **Demographic Parity**: Tracks distribution fairness across protected attributes

### 📊 Interactive Gradio UI
- **Search tab**: Natural language query, 6 adjustable scoring sliders, real-time re-ranking
- **Analytics tab**: Fairness metrics dashboard, score distribution histogram, candidate breakdown
- **About tab**: System documentation

---

## Hackathon CLI (batch ranking)

### Run the Pipeline

```bash
# Interactive mode (default) — prompts for a query, runs full agentic pipeline
# Planner → Executor → Reflector (uses LLM for deep understanding)
python rank.py

# Single query via CLI arg — same full agentic pipeline
python rank.py --query "senior software engineer with python and aws in bangalore"

# Batch mode — runs 20 pre-defined strategic queries (no LLM needed, turbo mode)
python rank.py --batch

# Validate output format
python validate_submission.py submission.csv

# View results
cat submission.csv | head -20
```

## 📊 Results & Performance

### Submission Quality (100K Profile Index)
| Metric | Value |
|--------|-------|
| **Pipeline time** | ~56.5s (20 queries × 50 results, CPU-only) |
| **Unique candidates matched** | 800+ (from 20 queries) |
| **Score range** | 0.7033 – 0.8657 |
| **Score monotonicity** | 100% non-increasing ✓ |
| **Unique reasoning** | 100/100 unique narratives ✓ |
| **Honeypots detected** | 11 (×0.15 penalty applied) |

### Test Coverage
| Suite | Tests | Status |
|-------|-------|--------|
| Unit + integration tests | 456 | ✅ 100% passing |
| End-to-end | 35 steps | ✅ 100% passing |
| Submission validation | — | ✅ VALID |

### Performance Budget
| Operation | Time | Notes |
|-----------|------|-------|
| Model loading (first run) | ~5-8s | Cached in HF cache |
| FAISS search (100K × 384d) | ~50ms | HNSW index |
| BM25 search (100K docs) | ~200ms | Partitioned with argpartition |
| Cross-encoder (50 candidates) | ~1.2s | Batch inference |
| Full pipeline (17 queries) | ~56s | 100K profile index |
| UI re-ranking | <100ms | Client-side only |

---

## 🧩 File Structure

```
📦 india-runs
├── 📄 rank.py                          # 🏁 Main pipeline: interactive, --query, --batch
├── 📄 validate_submission.py           # ✅ CSV format verifier
├── 📄 submission.csv                   # 📊 Generated output (100 rows)
├── 📄 submission_metadata.yaml         # 🏆 Hackathon portal metadata
│
├── 📁 src/
│   ├── 📁 agents/                      # 🧠 LangGraph agentic workflow
│   │   ├── orchestrator.py             # State machine, nodes, response builder
│   │   ├── planner.py                  # LLM + rule-based query parsing
│   │   ├── executor.py                 # Hybrid search → rerank → score pipeline
│   │   ├── reflector.py                # Quality evaluation + re-plan decision
│   │   └── prompts.py                  # LLM system prompts
│   │
│   ├── 📁 search/                      # 🔍 Retrieval layer
│   │   ├── hybrid.py                   # FAISS + BM25 + RRF fusion
│   │   ├── vector_search.py            # FAISS HNSW vector index
│   │   ├── bm25_search.py              # BM25 Okapi keyword index
│   │   ├── reranker.py                 # Cross-encoder reranking
│   │   └── filters.py                  # Pre-search structural filters
│   │
│   ├── 📁 matching/                    # 📐 Scoring engine
│   │   ├── scorer.py                   # Weighted 10-dim scoring + UI sliders
│   │   ├── skill_matcher.py            # 4-strategy: exact → normalized → alias → fuzzy
│   │   ├── experience_matcher.py       # YOE scoring with deficit/excess penalties
│   │   ├── behavioral_scorer.py        # 20+ signals, career trajectory, proficiency
│   │   └── confidence.py               # Score variance confidence
│   │
│   ├── 📁 ranking/                     # 🏅 Plackett-Luce listwise ranking
│   │   └── listwise_ranker.py          # Tournament ranking with LLM group judge
│   │
│   ├── 📁 language/                    # 🌐 Multilingual NLP
│   │   ├── multilingual.py             # 50+ language embeddings
│   │   ├── code_mixed.py               # Hinglish detection + NER + transliteration
│   │   ├── detector.py                 # Language identification (langdetect)
│   │   └── translator.py               # Google Translate + MBart-50 fallback
│   │
│   ├── 📁 fairness/                    # 🛡️ Bias mitigation
│   │   ├── anonymizer.py               # PII + style anonymization
│   │   ├── bias_detector.py            # 4-dimension bias analysis
│   │   └── metrics.py                  # DIR, demographic parity, language bias
│   │
│   ├── 📁 rationale/                   # 💬 Explanation generation
│   │   ├── generator.py                # LLM + template-based rationale
│   │   └── validator.py                # Quality validation
│   │
│   ├── 📁 extraction/                  # 🔧 Field extraction
│   │   ├── pipeline.py                 # Orchestrator
│   │   ├── title.py, company.py        # Cross-field extractors
│   │   ├── seniority.py, domain.py     # Seniority & domain inference
│   │   └── experience_years.py         # YOE from career history
│   │
│   ├── 📁 ingestion/                   # 📥 Data ingestion
│   │   ├── parser.py                   # JSONL/CSV/DOCX streaming parser
│   │   ├── normalizer.py               # Redrob API → Profile mapping
│   │   └── quality_scorer.py           # 0-1 data quality score
│   │
│   ├── 📁 core/                        # ⚙️ Core infrastructure
│   │   ├── config.py                   # Settings, YAML config, LLM factory
│   │   ├── models.py                   # 30+ Pydantic models
│   │   ├── constants.py                # 12 languages, 45 companies, 20 cities
│   │   └── profile_store.py            # Lazy-load profile cache + offset index
│   │
│   ├── 📁 api/                         # 🌐 FastAPI endpoints
│   │   ├── routes/search.py, profiles.py, ingest.py, health.py
│   │   └── middleware/logging.py, validation.py
│   │
│   ├── 📁 ui/                          # 🎨 Gradio dashboard
│   │   ├── app.py                      # 3-tab app, search handler, re-rank
│   │   ├── components.py               # Cards, score bars, radar charts, analytics
│   │   └── styles.css                  # Dark mode, glassmorphic design
│   │
│   ├── 📁 evaluation/                  # 📈 Metrics computation
│   │   └── metrics.py                  # P@k, R@k, MRR, nDCG
│   │
│   └── 📁 data/                        # 📦 Data loading utilities
│
├── 📁 scripts/                         # 🛠️ Utility scripts
│   ├── build_indexes.py                # FAISS + BM25 index builder
│   ├── evaluate.py                     # Evaluation runner
│   ├── generate_ground_truth.py        # Ground truth generator
│   └── e2e_test.py                     # 35-step end-to-end validation
│
├── 📁 tests/                           # 🧪 456 test suite
│   ├── test_agents/                    # Planner, reflector tests
│   ├── test_search/                    # FAISS, BM25, hybrid, filters
│   ├── test_matching/                  # Scorer, skill matcher, experience
│   └── test_ingestion/                 # Parser, normalizer, quality
│
├── 📁 configs/                         # ⚙️ YAML configuration
│   ├── settings.yaml                   # Search, agent, model parameters
│   ├── scoring_weights.yaml            # 10-dim internal + 6-dim slider weights
│   └── models.yaml                     # Model names, dimensions, devices
│
├── 📁 docs/                            # 📚 Documentation
│   ├── architecture.md                 # System architecture deep-dive
│   ├── api.md                          # API endpoint reference
│   └── deployment.md                   # Local, Docker, HF Spaces, Railway
│
├── 📄 PRD.md                           # Product Requirements Document (25 sections)
├── 📄 IMPLEMENTATION_PLAN.md           # 14-module execution blueprint
├── 📄 HACKATHON_WIN_PLAN.md            # Judging criteria alignment plan
├── 📄 pyproject.toml                   # Python project config + dependencies
├── 📄 Dockerfile                       # Container build
├── 📄 docker-compose.yml               # Multi-service orchestration
└── 📄 .env.example                     # Environment template
```

---

## 🛠️ Configuration

### Environment Variables (`.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai`, `gemini`, or `ollama` |
| `OPENAI_API_KEY` | — | OpenAI API key (if provider=openai) |
| `GEMINI_API_KEY` | — | Google API key (if provider=gemini) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MAX_REPLAN_CYCLES` | `1` | Max agent re-plan iterations (this, not `settings.yaml`, is what the orchestrator reads) |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformer for indexing and queries (must match the index) |
| `CROSS_ENCODER_ENABLED` | `true` | `false` skips the reranker model; rerank uses hybrid scores |
| `INGESTED_PROFILES_PATH` | `data/profiles/ingested.jsonl` | Store for profiles added via `/api/v1/ingest` |

### Scoring Weights (`configs/scoring_weights.yaml`)
Adjust the 10 internal scoring dimensions or the 6 UI slider dimensions. Customize per role.

### Search / Agent Parameters (`configs/scoring_weights.yaml`)
- `rrf_k`: RRF constant for hybrid fusion (default: 60)
- `min_good_matches_for_pass`: good matches needed before the reflector stops re-planning (default: 4)
- The re-plan limit comes from the `MAX_REPLAN_CYCLES` env var (default: 1).

`configs/settings.yaml` is only read for the translation model names; its `search:` and `agent:`
blocks are not used by the code.

---

## 🏆 Hackathon

- **Event**: [India Runs by Redrob AI](https://hack2skill.com/event/india_runs)
- **Track**: Track 1 — Data & AI Challenge (Intelligent Candidate Discovery)
- **Prize Pool**: ₹10 Lakhs (Grand Champion: ₹2,00,000)
- **Team**: Atlas — Nikhil Choudhary ([me@nikhilchoudhary.dev](mailto:me@nikhilchoudhary.dev))
- **GitHub**: [https://github.com/MrNK2107/India-Runs](https://github.com/MrNK2107/India-Runs)
- **Deadline**: June 28, 2026
- **Grand Finale**: July 22, 2026

---

## 📈 What Judges Will See

### During Demo
1. **Run the Pipeline** — `python rank.py` (interactive) or `python rank.py --batch` generates 100-row submission in ~56s
2. **Launch the UI** — `python src/ui/app.py` opens interactive Gradio dashboard
3. **Search** — Paste natural language queries, watch results with scored breakdowns
4. **Adjust Sliders** — Fine-tune scoring weights, see instant re-ranking
5. **Analytics** — Explore fairness metrics and score distributions
6. **Validate** — `python validate_submission.py submission.csv` confirms format

### Judging Criteria Alignment
| Criteria | Weight | How We Address It |
|----------|--------|-------------------|
| **Technical Execution** | 25% | 456 tests, 35 e2e steps, production-grade architecture |
| **Presentation** | 25% | Gradio UI, API docs, spectacular README, demo-ready |
| **Innovation** | 25% | Agentic LangGraph, Plackett-Luce, honeypot detection, TinT prompting |
| **Real-world Impact** | 25% | 100K profiles, 20+ signals, fairness auditing, multilingual |

---

## License (from the original README)

Built for India Runs by Redrob AI — Track 1: Data & AI Challenge.

---

