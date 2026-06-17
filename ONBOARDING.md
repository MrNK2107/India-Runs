# 🚀 India Runs — Welcome Aboard!

Hey there, new teammate! Welcome to the **India Runs** project — an AI-powered candidate ranking system built for the **Redrob AI Hackathon** (prize pool: ₹10 Lakhs!). We're competing in Track 1: Data & AI Challenge.

This doc is your friendly guide to:
- **The Problem** — What the hell are we even solving?
- **The PRD** — The grand plan (the *what* and *why*)
- **The Codebase** — How we actually built it (the *how*)

Grab a chai ☕ and let's dive in.

---

## Table of Contents
1. [The Problem — Why This Exists](#1-the-problem--why-this-exists)
2. [The PRD — The Grand Plan](#2-the-prd--the-grand-plan)
3. [The Codebase — The How](#3-the-codebase--the-how)
4. [Key Files to Know](#4-key-files-to-know)
5. [How It All Comes Together](#5-how-it-all-comes-together)
6. [What's Done vs What's Coming](#6-whats-done-vs-whats-coming)
7. [Quick Start (if you want to run it)](#7-quick-start-if-you-want-to-run-it)
8. [Glossary](#8-glossary)

---

## 1. The Problem — Why This Exists

### The Short Version

**Keyword-based resume matching is broken.** And it's hurting both recruiters and candidates.

### Imagine This...

You're a recruiter named Priya at a Bengaluru startup. You need a **DevOps engineer** who's scaled infrastructure at a mid-size company. You paste "DevOps, Kubernetes, CI/CD, 3+ years" into your ATS (Applicant Tracking System).

The system spits back 10 resumes. All have the exact keywords. Perfect, right?

**Wrong.** Here's who you missed:
- A **site reliability engineer** at Swiggy who manages 500+ microservices — but wrote "incident response" instead of "DevOps"
- A **platform engineer** who built deployment pipelines for 200 engineers — but listed "build automation" not "CI/CD"
- A **cloud architect** with 8 years AWS experience — but her resume says "infrastructure optimization" not "Kubernetes"

Meanwhile, here's who made the cut:
- A candidate who copy-pasted "DevOps, Kubernetes, CI/CD" into their resume but never actually used them

This is the **keyword matching trap**. And it's everywhere.

### The Indian Market Makes It Worse

| Challenge | What It Means |
|-----------|--------------|
| **Multilingual profiles** | Resumes in Hindi, Tamil, Telugu, Bengali — keyword search doesn't work across languages |
| **Unstandardized data** | Profiles from LinkedIn, Naukri, AngelList, GitHub — all different formats, all messy |
| **Passive talent ~70%** | Most good engineers aren't job-hunting. They don't have "looking for work" on their profile |
| **False positives** | "Marketing Manager" can look like "Product Manager" to a dumb algorithm |
| **Bias** | AI systems learn to favor IIT grads, Bangalore addresses, and male-coded names if you're not careful |

### What Redrob Specifically Needs

Redrob has **700M+ profiles**. They need a system that:
- Can search **700M profiles** in milliseconds
- Handles **30+ Indian languages** natively
- Understands **Indian hiring context** (tier-2 city salary norms, local career paths)
- Provides **explanations** — not just a ranked list
- Runs **fast** (< 50ms for production)

### The One-Sentence Problem Statement

> Build an AI recruiter that reads every resume intelligently, understands *meaning* not just keywords, combines multiple signals (skills, experience, certifications, engagement), and hands back a ranked shortlist with clear reasons — in seconds.

---

## 2. The PRD — The Grand Plan

Our `PRD.md` (v2.1, 2,319 lines, 25 sections) is the **constitution** of this project. It's huge, but here's what matters.

### The Vision

We're building an **Intelligent Candidate Discovery System** with 5 layers:

```
Layer 1: Input & Parsing   — Eat messy resumes, digest into structured profiles
Layer 2: Semantic Search   — Understand meaning, not just words (vector embeddings)
Layer 3: Multi-Signal Scoring — Score across 6 dimensions (skills, experience, etc.)
Layer 4: Agentic Workflow  — Plan → Execute → Reflect → Re-plan (like a human recruiter)
Layer 5: Explainable Output — Show *why* each candidate ranks where they do
```

### Key Differentiators (Why We Win)

1. **Explainability** — Existing tools (HireVue, Eightfold, Workday) rank candidates but don't say *why*. We generate plain-English rationales for every match.
2. **Semantic Understanding** — We use modern LLMs as the brain, not just keyword extraction.
3. **Agentic AI** — Our system reflects on its own results and re-plans if it's not satisfied (like a good recruiter who tries different search terms).
4. **Fairness-First** — Built-in bias detection across 4 dimensions (name, language, location, university).
5. **Hybrid Search** — BM25 (keyword) + FAISS (vector) + RRF fusion + Cross-Encoder reranking — beats pure vector search by ~25%.

### Scoring Formula

The final score for a candidate is a weighted combination:

```
overall = 0.25 × semantic_similarity
        + 0.15 × keyword_match
        + 0.30 × skill_match        ← Skills matter most
        + 0.15 × experience_match
        + 0.05 × location_match
        + 0.05 × education_match
        + 0.05 × cross_encoder_score
```

These weights live in `configs/scoring_weights.yaml` — NOT hardcoded. You can tweak them without touching Python code.

### Success Metrics

| Metric | Target | Why It Matters |
|--------|--------|---------------|
| Precision@10 | ≥ 0.85 | 85% of top 10 should be good hires |
| Recall@50 | ≥ 0.90 | Find 90% of all good candidates |
| Cross-lingual MRR | ≥ 0.75 | Works across Hindi, Tamil, etc. |
| Response time | < 2s | A recruiter won't wait longer |
| Disparate Impact Ratio | ≥ 0.80 | No demographic group gets screwed |
| Test coverage | ≥ 80% | We ship with confidence |

### The Competition (How We Stack Up)

| Tool | Strengths | Their Gap |
|------|-----------|-----------|
| **HireVue** | Video interview AI | Needs video — useless without it |
| **Workday** | Deep ATS integration | Still keyword-heavy under the hood |
| **Eightfold** | Trained on billions of profiles | Black box — no explainability |
| **Greenhouse** | Easy to use | Dumb keyword matching |

**Our advantage:** We combine semantic search + explainability + fairness + agentic reflection. Nobody does all four.

---

## 3. The Codebase — The How

### Project Structure (The Bird's Eye View)

```
india-runs/
├── configs/                    # YAML config files (weights, models, settings)
├── src/
│   ├── core/                   # Foundation: models, config, constants
│   ├── ingestion/              # Eat profiles (JSONL/JSON/CSV/DOCX), normalize them
│   ├── language/               # Handle Hindi/Tamil/Hinglish etc.
│   ├── search/                 # FAISS (vector) + BM25 (keyword) + hybrid fusion
│   ├── matching/               # Skill matching, experience matching, scoring
│   ├── agents/                 # LangGraph state machine (the AI brain)
│   ├── rationale/              # Generate "why this candidate" explanations
│   ├── fairness/               # Bias detection & fairness metrics
│   ├── api/                    # FastAPI endpoints
│   └── ui/                     # Gradio frontend
├── scripts/                    # Build indexes, run evaluation
├── tests/                      # 87 tests, all passing 🟢
├── docs/                       # Architecture, API, deployment docs
├── data/                       # Profile data (100K real Redrob candidates!)
├── PRD.md                      # The master plan
├── IMPLEMENTATION_PLAN.md      # Detailed execution blueprint
└── ONBOARDING.md               # 👈 You are here
```

### Layer 1: Core Foundation (`src/core/`)

The bedrock of the entire system.

**`models.py`** — 30 Pydantic models + 7 enums. Everything in the system flows through these types.

The hierarchy is:
```
Profile
├── PersonalInfo (name, location, languages)
├── ProfessionalInfo (title, company, exp years, industry)
├── Skill[] (name, category, proficiency, years)
├── WorkExperience[] (title, company, dates, description)
├── Education[] (institution, degree, field, GPA)
├── Signals (passive, github_activity, certifications)
└── ProfileMetadata (language, quality score, embedding ID)
```

Then there's the pipeline chain:
```
JobQuery → ParsedQuery → SearchRequest → MatchResult → SearchResponse
           (structured)                   (per candidate)   (final output)
```

**`config.py`** — Reads YAML configs with `@lru_cache`, creates the LLM client via a factory (`get_llm_client()`). Supports OpenAI, Gemini, and local Ollama. Switch providers by changing one env var.

**`constants.py`** — 12 Indian languages, 45 Indian companies, 20 Indian cities, 20 universities. This is our "Indian context" database.

### Layer 2: Ingestion (`src/ingestion/`)

How we eat 100K profiles without choking.

**`parser.py`** — `ProfileParser` that streams JSONL **line by line** (487MB file, never loads into memory). Also handles JSON, gzip, DOCX (zipfile + XML), and CSV. Skips noisy profiles gracefully.

**`normalizer.py`** — `normalize_redrob()` maps the Redrob API schema (8 top-level fields) to our internal Profile model (30+ fields). Auto-infers skill categories via keyword matching — "Python" → programming_language, "React" → framework, "Kubernetes" → tool.

**`quality_scorer.py`** — `compute_data_quality_score()` gives each profile a 0-1 score based on how complete it is. Missing name? Penalty. No skills? Penalty. Encoding artifacts? Penalty.

### Layer 3: Language Pipeline (`src/language/`)

The multilingual superpower.

**`detector.py`** — Uses `langdetect` to identify if a profile is in Hindi, Tamil, Hinglish, etc.

**`code_mixed.py`** — This is *fun*. `CodeMixedProcessor` detects **Hinglish** (Hindi+English mix) using 3 strategies:
1. Checking for Devanagari characters
2. Looking for 185 Hinglish keywords ("hai", "nahi", "karo", "achha")
3. Counting Latin vs non-Latin words

It also has a **transliteration map** of 80+ words (kaam → work, baat → talk, achha → good) and regex-based NER fallback for skills and companies.

**`translator.py`** — `TranslationPipeline` using `deep-translator` (Google Translate, free, no API key). Previously had Helsinki-NLP opus-mt models for 9 Indian languages (~300MB each) which were impractical. All 100K profiles are in English, but non-English queries are handled via Google Translate for free.

**`multilingual.py`** — `MultilingualEmbedder` using `paraphrase-multilingual-MiniLM-L12-v2` (384-dim). Supports 50+ languages. Normalized embeddings so cosine similarity = dot product.

### Layer 4: Search Engine (`src/search/`)

The hybrid retrieval monster.

**`vector_search.py`** — FAISS `IndexFlatIP` (inner product on 384-dim normalized vectors). Build, search, save, load with JSON id_map for profile-to-vector lookups.

**`bm25_search.py`** — Good old BM25Okapi from `rank_bm25`. Case-insensitive tokenization.

**`hybrid.py`** — `HybridSearch` runs FAISS and BM25 **in parallel**, then fuses results using **Reciprocal Rank Fusion (RRF)**.

```python
# The RRF formula (from src/search/hybrid.py:43)
scores[doc_id] += 1.0 / (k + rank)
```

The `k` parameter (default 60) controls how much high ranks dominate. Lower k = more aggressive fusion.

**`reranker.py`** — `CrossEncoderReranker` using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Takes top-50 hybrid results, reranks them with a more expensive but more accurate model. Configurable timeout (default 500ms) so it doesn't stall the pipeline.

**`filters.py`** — `SearchFilter` applies hard structural filters BEFORE the expensive search. Location, experience range, company include/exclude. This is called **Scoped Pre-Search Retrieval** in the PRD.

### Layer 5: Matching & Scoring (`src/matching/`)

Where candidates get their score.

**`skill_matcher.py`** — `SkillMatcher` uses 4 strategies in order:
1. **Exact match** — "Python" == "Python"
2. **Normalized match** — "python" == "Python"
3. **Alias match** — "k8s" == "kubernetes", "react.js" == "react", "ml" == "machine learning" (32 curated aliases)
4. **Fuzzy match** — SequenceMatcher for typos

Proficiency scoring: beginner=0.25, intermediate=0.50, advanced=0.75, expert=1.0

**`experience_matcher.py`** — `ExperienceMatcher` scores years (deficit penalized heavily, excess gets diminishing returns capped at 2x). Industry match: exact=1.0, semantically similar=0.6 (like "fintech" ↔ "banking"), different=0.3.

**`scorer.py`** — `CandidateScorer` combines 7 dimensions using weights from YAML. **Renormalizes** when some dimensions are null (e.g., no education data). Confidence = 1 - std_dev of scores (more agreement = higher confidence).

### Layer 6: The AI Brain — Agentic Workflow (`src/agents/`)

This is where it gets **cool**. We use **LangGraph** to create a state machine that mimics how a good recruiter works:

```
     Plan ──→ Execute ──→ Reflect ──→ Re-plan ──→ Plan (max 3 cycles)
                                                          │
                                                     [Good enough?]
                                                          │
                                                     Generate Output
```

**`planner.py`** — `PlannerAgent` takes a job query like "find me a senior DevOps engineer in Bangalore who knows AWS and Kubernetes" and extracts structured search parameters using an LLM. Supports **TinT (Translate-in-Thought)** for code-mixed queries (tells the LLM to internally translate Hindi to English before parsing). Has fallback keyword extraction if no LLM is available.

**`executor.py`** — `ExecutorAgent` runs the full pipeline: hybrid search → structural filters → cross-encoder reranker → candidate scorer → ranked `MatchResult` list.

**`reflector.py`** — `ReflectorAgent` evaluates results using LLM or a heuristic threshold. If < 8/10 matches are "good", it triggers a **re-plan** with relaxed criteria (broader location, lower experience bar, more skill aliases).

**`orchestrator.py`** — `Orchestrator` builds the LangGraph `StateGraph` with 4 nodes (Plan → Execute → Reflect → Generate Rationale), conditional edges, and max 3 replan cycles.

### Layer 7: Rationale Generation (`src/rationale/`)

The "why" behind every match.

**`generator.py`** — `RationaleGenerator` uses an LLM to write a human-readable explanation for why a candidate ranks where they do. Falls back to a template if LLM is unavailable.

Output includes:
- **Summary** — One-paragraph overview
- **Strengths** — What makes them great for this role
- **Gaps** — What they're missing
- **Skill details** — Per-skill matching evidence
- **Experience analysis** — Career trajectory assessment
- **Recommendation** — STRONG / GOOD / POTENTIAL / WEAK

**`validator.py`** — Checks rationale quality: summary length, strength presence, valid recommendation. `validate_batch()` for bulk stats.

### Layer 8: Fairness & Bias (`src/fairness/`)

Because we're not Amazon (who had to shut down their biased AI hiring tool).

**`bias_detector.py`** — `BiasDetector` checks 4 dimensions:
1. **Name bias** — Groups candidates by first-character of name, checks if certain names get higher scores
2. **Language bias** — Compares English vs non-English profile scores
3. **Location bias** — Tier-1 cities (Bangalore, Mumbai, Delhi) vs tier-2/3 cities
4. **University bias** — IIT/NIT/BITS vs other institutions

Each check returns `detected` (boolean), `observations` (human-readable), and `details` (raw numbers).

**`metrics.py`** — Computes `demographic_parity()`, `disparate_impact_ratio()` with the **4/5ths rule** (if a group's selection rate is < 80% of the highest group's rate, that's bias), and aggregates everything into a fairness dashboard.

### Layer 9: API (`src/api/`)

FastAPI with 4 endpoints:

| Endpoint | Method | What It Does |
|----------|--------|-------------|
| `/api/v1/search` | POST | Main search — takes query + filters, returns ranked candidates |
| `/api/v1/profiles/{id}` | GET | Get one profile |
| `/api/v1/profiles` | GET | Paginated profile list |
| `/api/v1/ingest` | POST | Upload profiles |
| `/api/v1/health` | GET | Status check |

Middleware handles logging (`RequestLoggingMiddleware`) and validation (`InputValidationMiddleware`, rejects > 10MB).

### Layer 10: UI (`src/ui/`)

Gradio app with 3 tabs:

1. **Search** — Query input + filters + example chips + ranked results with score badges, skill chips, radar charts
2. **Analytics** — Fairness metrics dashboard with bias alerts
3. **About** — Architecture diagram + tech stack

### Config System (`configs/`)

Everything is externalized to YAML:

- **`settings.yaml`** — App name, version, DB URLs, model names, search params, agent limits
- **`scoring_weights.yaml`** — 7 scoring weights, proficiency scores, skill importance, RRF k
- **`models.yaml`** — Model names, dimensions, devices

These support **env var interpolation** — `${DATABASE_URL}`, `${LLM_PROVIDER:-openai}` — so you can configure without touching code.

---

## 4. Key Files to Know

### Must-Read (in order)

| File | Why |
|------|-----|
| `PRD.md` | The entire product spec — 25 sections, 2,319 lines |
| `IMPLEMENTATION_PLAN.md` | 14-module execution blueprint with checklists |
| `src/core/models.py` | All data types — understand this and you understand the data flow |
| `src/search/hybrid.py` | The heart of search — FAISS + BM25 + RRF in 44 lines |
| `src/agents/orchestrator.py` | The LangGraph state machine — Plan → Execute → Reflect → Re-plan |
| `src/matching/scorer.py` | The scoring formula — 58 lines of pure math |
| `configs/scoring_weights.yaml` | The weights that control everything — 23 lines |
| `src/fairness/bias_detector.py` | How we prevent the system from being biased |

### Fun Files to Explore

| File | Why It's Interesting |
|------|---------------------|
| `src/language/code_mixed.py` | Has a 150+ word Hinglish keyword dictionary and transliteration map |
| `src/matching/skill_matcher.py` | Has 32 curated skill aliases — "k8s" → "kubernetes" |
| `src/ingestion/parser.py` | Streams 487MB without loading into memory |
| `src/fairness/bias_detector.py` | Checks if IIT grads get better scores (spoiler: they shouldn't) |
| `src/core/constants.py` | 45 Indian companies, 20 cities, 12 languages, 20 universities |

---

## 5. How It All Comes Together

### The Data Flow

```
📄 Profiles (JSONL, 100K profiles, 487MB)
    │
    ▼
🔄 Parser (streams line-by-line) → Normalizer (Redrob → internal model)
    │
    ▼
🧠 MultilingualEmbedder (384-dim vectors) + BM25 (keyword tokens)
    │
    ▼
📦 FAISS Index (vector search) + BM25 Index (keyword search)
                                        │
                                    [User types a query]
                                        │
                                        ▼
                                   🔮 PLANNER AGENT
                                   (LLM extracts skills, experience,
                                    location from natural language)
                                        │
                                        ▼
                                   🔍 EXECUTOR AGENT
                                   ├── Hybrid Search (FAISS + BM25 + RRF in parallel)
                                   ├── Structural Filters (location, exp, companies)
                                   ├── Cross-Encoder Reranker (smart reranking)
                                   └── Candidate Scorer (weighted combination)
                                        │
                                        ▼
                                   🪞 REFLECTOR AGENT
                                   ├── "Are these results good?"
                                   ├── YES → Generate Rationale → Output
                                   └── NO  → Re-plan with relaxed criteria → 🔄
                                        │
                                        ▼
                                   📝 RATIONALE GENERATOR
                                   (LLM writes "why this candidate" for each)
                                        │
                                        ▼
                                   🖥️ RESPONSE
                                   (Ranked list + scores + explanations)
```

### The Scoring Pipeline (Inside Executor)

```
Query ──→ Vector Search (FAISS) ──┐
Query ──→ BM25 Search ────────────┤
                                   ├──→ RRF Fusion ──→ Filters ──→ Reranker ──→ Scorer ──→ Results
                                                            ↑               ↑
                                                   Hard cut on       Weights from
                                                   location, exp     YAML config
                                                   companies
```

---

## 6. What's Done vs What's Coming

### ✅ Implemented (All 15 Phases)
- Core models (30 Pydantic models), config system, constants
- Profile ingestion: parser, normalizer, quality scorer
- Language pipeline: detector, translator stub, multilingual embedder, code-mixed processor
- Search engine: FAISS, BM25, hybrid RRF, cross-encoder reranker, structural filters
- Matching engine: skill matcher (32 aliases), experience matcher, weighted scorer
- LangGraph agentic workflow: Planner, Executor, Reflector, Orchestrator (Plan → Execute → Reflect → Re-plan, max 3 cycles)
- Rationale generator + validator
- Bias detector (4 dimensions) + fairness metrics (DIR, demographic parity)
- FastAPI (4 endpoints) + middleware
- Gradio UI (3 tabs: Search, Analytics, About)
- Scripts: index builder, evaluator, data generator
- 87 tests (pytest, all passing 🟢), ruff clean
- Full documentation: README, architecture, API, deployment

### 🔜 Planned But Not Yet Built
These are in the PRD but NOT in current code:
- **Plackett-Luce listwise tournament ranking** — candidates compete head-to-head
- **Feedback loop & RLHF** — recruiters accept/reject, system learns and re-weights
- **PII Anonymizer** — strip names/universities/locations before LLM evaluation
- **12-dimensional YAML rationale** (currently 6 dimensions)
- **Interactive scoring slider UI** — recruiters adjust weights visually
- **CI pipeline** — GitHub Actions
- **Rate limiting middleware**, metrics tracking, structured JSON logging

### 🐛 Known Quirks
- ~~Translation pipeline used heavy opus-mt models~~ **FIXED**: now uses `deep-translator` (Google Translate, free, no API key)
- ~~Constants were too limited for "Indian market" claims~~ **FIXED**: expanded to 120+ cities, 60+ universities, 120+ companies
- ~~Vector embeddings not used in scoring (semantic_similarity + keyword_match both used same RRF rank)~~ **FIXED**: semantic_similarity = actual FAISS cosine similarity, keyword_match = normalized BM25 score
- Some pip dependency conflicts with unrelated supabase packages
- The full 100K profiles (487MB) are NOT in git — you need the data file separately

---

## 7. Quick Start (if you want to run it)

```bash
# Install
pip install -e ".[dev]"

# Build indexes (FAISS + BM25)
python scripts/build_indexes.py

# Run evaluation
python scripts/evaluate.py

# Start the API
uvicorn src.main:app --reload

# Or start the Gradio UI
python src/ui/app.py

# Run tests
pytest tests/ -v --cov=src
```

See `README.md` for full setup.

---

## 8. Glossary

| Term | Meaning |
|------|---------|
| **ATS** | Applicant Tracking System — software recruiters use to manage candidates |
| **BM25** | Keyword search algorithm (like TF-IDF but smarter) |
| **Cross-Encoder** | A model that compares two texts at once (slower but more accurate) |
| **DIR** | Disparate Impact Ratio — fairness metric (should be ≥ 0.80) |
| **FAISS** | Facebook AI's vector search library — finds similar vectors fast |
| **Hinglish** | Hindi + English code-mixed language ("Yeh kaam achha hai") |
| **LangGraph** | A library for building state machines with LLMs |
| **LLM** | Large Language Model (GPT, Claude, Gemini) |
| **MRR** | Mean Reciprocal Rank — search quality metric |
| **nDCG** | Normalized Discounted Cumulative Gain — ranking quality metric |
| **P@k** | Precision at k — how many of top k results are relevant |
| **Plackett-Luce** | A ranking model where candidates "compete" in tournament rounds |
| **RLHF** | Reinforcement Learning from Human Feedback — learn from recruiter decisions |
| **RRF** | Reciprocal Rank Fusion — combines multiple search results into one ranking |
| **Semantic Search** | Search based on meaning (vectors) not keywords |
| **sentence-transformers** | Library that converts text into meaning vectors |
| **TinT** | Translate-in-Thought — LLM internally translates before processing |

---

**Welcome to the team! 🎉** 

The best place to start is `src/core/models.py` (understand the data types), then `src/search/hybrid.py` (understand the search), then `src/agents/orchestrator.py` (understand the AI brain).

If something's confusing, ask. We were all new here once.
