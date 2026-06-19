# India Runs — Engineer Onboarding Guide

> **Hackathon:** India Runs by Redrob AI — Track 1: Data & AI Challenge
> **Prize Pool:** ₹10 Lakhs
> **Python:** 3.12 | **Tests:** 149 passing | **mypy:** strict clean

---

## Table of Contents

1. [Codebase Map — Every File, What It Does](#1-codebase-map)
2. [Data Flow — End to End](#2-data-flow)
3. [Layer-by-Layer Deep Dive](#3-layer-by-layer-deep-dive)
4. [Configuration Reference](#4-configuration-reference)
5. [API Reference](#5-api-reference)
6. [Testing Strategy](#6-testing-strategy)
7. [How to Extend](#7-how-to-extend)
8. [Debugging & Performance](#8-debugging--performance)
9. [PRD-to-Code Mapping](#9-prd-to-code-mapping)
10. [Glossary](#10-glossary)

---

## 1. Codebase Map

```
india-runs/
├── configs/                          # YAML config — change weights without touching code
│   ├── settings.yaml                 #   App, DB, model names, search params
│   ├── scoring_weights.yaml          #   7 scoring dimensions + 6 slider dims
│   └── models.yaml                   #   Model names, dimensions, devices
├── src/
│   ├── core/
│   │   ├── models.py                 #   30 Pydantic models + 7 enums (THE source of truth)
│   │   ├── config.py                 #   YAML loader, LLM factory, Settings class
│   │   ├── constants.py              #   45 Indian companies, 20 cities, 12 languages
│   │   └── profile_store.py          #   Lazy-load profiles from JSONL by offset
│   ├── extraction/                   #   ★ NEW — Smart cross-field extraction
│   │   ├── career_history_utils.py   #   latest_role(), compute_years_from_dates()
│   │   ├── seniority.py              #   Title → history → years fallback
│   │   ├── experience_years.py       #   Direct → computed dates → regex fallback
│   │   ├── title.py                  #   Direct → history → headline parsing
│   │   ├── company.py                #   Direct → history
│   │   ├── domain.py                 #   Industry from 4 signals
│   │   └── pipeline.py               #   FieldExtractorPipeline orchestrator
│   ├── ingestion/
│   │   ├── parser.py                 #   Stream JSONL line-by-line (487MB, no memory load)
│   │   ├── normalizer.py             #   normalize_redrob() → uses FieldExtractorPipeline
│   │   ├── extractor.py              #   LLM-based FieldExtractor for DOCX fallback
│   │   └── quality_scorer.py         #   compute_data_quality_score() 0-1
│   ├── language/
│   │   ├── detector.py               #   langdetect wrapper
│   │   ├── translator.py             #   deep-translator Google Translate (free, no key)
│   │   ├── code_mixed.py             #   Hinglish detector + 80-word transliteration map
│   │   └── multilingual.py           #   paraphrase-multilingual-MiniLM-L12-v2 wrapper
│   ├── search/
│   │   ├── vector_search.py          #   FAISS IndexFlatIP(384), save/load with id_map
│   │   ├── bm25_search.py            #   rank_bm25 BM25Okapi, tokenize=lower().split()
│   │   ├── hybrid.py                 #   FAISS + BM25 parallel → RRF fusion, SearchCache
│   │   ├── reranker.py               #   cross-encoder/ms-marco-MiniLM-L-6-v2, sigmoid
│   │   └── filters.py                #   SearchFilter: location, exp range, companies
│   ├── matching/
│   │   ├── skill_matcher.py          #   32 aliases, 4-strategy: exact→normalized→alias→fuzzy
│   │   ├── experience_matcher.py     #   Years scoring + industry match
│   │   ├── scorer.py                 #   Weighted 7-dim combination, slider support
│   │   └── confidence.py             #   Confidence from score variance
│   ├── agents/
│   │   ├── planner.py                #   LLM extracts structured params from NL query
│   │   ├── executor.py               #   Runs hybrid→filters→rerank→score pipeline
│   │   ├── reflector.py              #   LLM evaluates results, decides re-plan
│   │   ├── orchestrator.py           #   LangGraph StateGraph: Plan→Execute→Reflect→Re-plan
│   │   └── prompts.py                #   System prompts for planner + reflector LLMs
│   ├── rationale/
│   │   ├── generator.py              #   LLM writes "why this candidate", template fallback
│   │   └── validator.py              #   Checks rationale quality (length, strengths, recommendation)
│   ├── fairness/
│   │   ├── bias_detector.py          #   4 dimensions: name, language, location, university
│   │   └── metrics.py                #   demographic_parity(), disparate_impact_ratio()
│   ├── api/
│   │   ├── routes/
│   │   │   ├── search.py             #   POST /api/v1/search
│   │   │   ├── profiles.py           #   GET /api/v1/profiles, GET /api/v1/profiles/{id}
│   │   │   ├── ingest.py             #   POST /api/v1/ingest
│   │   │   └── health.py             #   GET /api/v1/health
│   │   └── middleware/
│   │       ├── logging.py            #   RequestLoggingMiddleware
│   │       └── validation.py         #   InputValidationMiddleware (rejects >10MB)
│   ├── ui/
│   │   ├── app.py                    #   Gradio app: Search, Analytics, About tabs
│   │   └── components.py             #   Score badges, radar charts, skill chips
│   ├── evaluation/
│   │   └── metrics.py                #   Compute precision@k, recall@k, nDCG, MRR
│   └── main.py                       #   FastAPI app factory + lifespan (loads models)
├── scripts/
│   ├── build_indexes.py              #   Read 100K profiles → normalize → embed → FAISS+BM25
│   ├── evaluate.py                   #   Run evaluation against ground truth
│   └── data_generator.py             #   Create synthetic profiles
├── tests/                            #   149 tests, all passing
│   ├── test_extraction/              #   63 tests for the new smart normalizer
│   ├── test_ingestion/               #   Parser, normalizer, quality scorer
│   ├── test_search/                  #   FAISS, BM25, hybrid, filters
│   ├── test_matching/                #   Skill matcher, experience, scorer
│   ├── test_agents/                  #   Planner fallback, reflector
│   ├── test_api/                     #   Search endpoint, bias detector
│   ├── test_language/                #   Detector, translator, code-mixed, embedder
│   ├── test_rationale/               #   Generator, validator
│   └── test_integration/             #   End-to-end, model existence
├── docs/
│   ├── architecture.md               #   System architecture overview
│   ├── api.md                        #   API documentation
│   ├── deployment.md                 #   Deployment guide
│   └── evaluation.md                 #   Evaluation methodology
├── data/
│   ├── profiles/candidates.jsonl     #   100K real Redrob profiles (487MB, not in git)
│   └── indexes/                      #   Built by build_indexes.py
├── PRD.md                            #   2,319 lines — the constitution
├── IMPLEMENTATION_PLAN.md            #   Execution blueprint
└── README.md                         #   5-min quick start
```

---

## 2. Data Flow

### 2.1 Ingestion Pipeline (Offline)

```
JSONL (100K lines)                  data/profiles/candidates.jsonl
    │
    ▼
ProfileParser.parse_jsonl_file()    src/ingestion/parser.py:17-40
    │  Streaming generator — one dict at a time, never loads 487MB
    ▼
normalize_redrob(raw_dict)          src/ingestion/normalizer.py:40-159
    │  Maps Redrob schema → Profile model
    │  Uses FieldExtractorPipeline for smart cross-field extraction
    ▼
Profile object                      src/core/models.py:138-148
    │  Pydantic model with nested PersonalInfo, ProfessionalInfo,
    │  Skill[], WorkExperience[], Education[], Signals, ProfileMetadata
    ▼
compute_data_quality_score()        src/ingestion/quality_scorer.py:19-46
    │  Filters out profiles with score < 0.3
    ▼
MultilingualEmbedder.embed()        src/language/multilingual.py:29-30
    │  paraphrase-multilingual-MiniLM-L12-v2 → 384-dim vector
    ▼
VectorSearch.build_index()          src/search/vector_search.py:18-21
    │  FAISS IndexFlatIP — adds all vectors
    │
    ├── FAISS index saved to data/indexes/faiss_index.bin
    └── id_map saved to data/indexes/faiss_id_map.json

BM25Search.build_index()            src/search/bm25_search.py:18-21
    │  BM25Okapi over _build_document_text() — concatenated profile fields
    │
    └── BM25 index saved to data/indexes/bm25_index.pkl
```

### 2.2 Search Flow (Online)

```
User Query: "Find a senior DevOps engineer in Bangalore"
    │
    ▼
POST /api/v1/search                 src/api/routes/search.py
    │
    ▼
Orchestrator.run()                  src/agents/orchestrator.py:127-148
    │  Checks if query ≤3 words → _turbo_run (skip agent loop)
    │  Otherwise → full LangGraph state machine
    │
    ├── PLANNER
    │   PlannerAgent.plan()          src/agents/planner.py
    │   LLM extracts: skills=[DevOps, Kubernetes], experience=5yrs, location=Bangalore
    │   Returns ParsedQuery (Pydantic)
    │
    ├── EXECUTOR
    │   ExecutorAgent.execute()      src/agents/executor.py:137-231
    │   │
    │   ├── HybridSearch.search()    src/search/hybrid.py:54-74
    │   │   │  embed_query() → FAISS search (parallel)
    │   │   │  BM25 search (parallel)
    │   │   │  RRF fusion: score += 1/(k + rank) for each result list
    │   │   │  Returns [(profile_id, rrF_score), ...]
    │   │   ▼
    │   ├── _apply_filters()          executor.py:260-281
    │   │   │  SearchFilter.passes() → location, exp range, companies
    │   │   │  Drops profiles that don't match hard criteria
    │   │   ▼
    │   ├── CrossEncoderReranker     src/search/reranker.py:34-65
    │   │   │  Scores (query, profile_raw_text) pairs
    │   │   │  Sigmoid-normalized output
    │   │   │  Configurable timeout_ms — skips if too slow
    │   │   ▼
    │   └── CandidateScorer          src/matching/scorer.py:24-79
    │       │  Computes 7 dimensions: semantic, keyword, skill, experience,
    │       │  location, education, cross_encoder
    │       │  Weighted combination: overall = Σ(weight × score) / Σ(weight)
    │       │  Returns MatchScores with confidence
    │       │
    │       └── Returns list[MatchResult], sorted by score
    │
    ├── REFLECTOR
    │   ReflectorAgent.reflect()     src/agents/reflector.py
    │   │  LLM evaluates: are these actually good matches?
    │   │  If < threshold → should_replan=True
    │   │
    │   ├── PASS → GENERATE RATIONALE
    │   │   RationaleGenerator generates "why this candidate" text
    │   │
    │   └── FAIL → RE-PLAN (max 3 cycles)
    │       PlannerAgent.replan() with relaxed params
    │
    ▼
SearchResponse                      src/core/models.py:307-314
    Ranked list + scores + rationale + metadata
```

### 2.3 Tracking a Real Request

Query: `"backend engineer python 3 years bangalore"`

1. **orchestrator.py:131** — `_is_simple_query()` detects 6 words → false
2. **orchestrator.py:134** — Full agent loop starts
3. **planner.py** — LLM parses to `ParsedQuery(required_skills=[{name:"python"}, {name:"backend"}], experience={min_years:3}, location={city:"Bangalore"})`
4. **executor.py:145** — `hybrid_search.search("python backend 3+ years experience Bangalore")`
5. **hybrid.py:59** — Query embedded to 384-dim vector
6. **hybrid.py:60** — FAISS searches 100K vectors → top 100 results
7. **hybrid.py:61** — BM25 searches 100K documents → top 100 results
8. **hybrid.py:72** — RRF fuses both lists → top 100 unique profile IDs
9. **executor.py:158** — `_apply_filters()` drops profiles not in Bangalore
10. **executor.py:160-166** — Top 20 go to cross-encoder reranker
11. **reranker.py:45** — Cross-encoder scores each (query, profile_text) pair
12. **executor.py:170-229** — For each reranked profile:
    - Skill match computed (exact→alias→fuzzy→raw_text scan)
    - 7 scores assembled → `CandidateScorer.compute_overall()`
    - `MatchResult` built with matched/missing skills
13. **executor.py:231** — Returns list[MatchResult] sorted by overall score
14. **reflector.py** — LLM confirms ≥8/10 are good → PASS
15. **orchestrator.py:253** — `_build_response()` → SearchResponse

**Total latency target: < 2s** (FAISS+BM25 ~50ms, cross-encoder ~500ms, LLM calls ~1s)

---

## 3. Layer-by-Layer Deep Dive

### 3.1 Core (`src/core/`)

#### `models.py` — 30 Pydantic Models, 7 Enums

Everything flows through these types. If you don't understand the models, you don't understand the system.

**Enums:**
- `ProfileSource` (line 10): linkedin, naukri, github, resume_pdf, career_page, manual, redrob
- `SkillCategory` (line 20): programming_language, framework, tool, soft_skill, domain_knowledge, certification
- `ProficiencyLevel` (line 29): beginner, intermediate, advanced, expert
- `SkillImportance` (line 36): required, preferred, nice_to_have
- `EmploymentType` (line 42): full_time, part_time, contract, freelance, student
- `MatchRecommendation` (line 50): strong_match, good_match, potential_match, weak_match
- `SearchMethod` (line 57): hybrid, vector_only, keyword_only

**Profile Hierarchy (lines 63-148):**
```
Profile
├── profile_id: str (UUID)
├── source: ProfileSource
├── raw_text: str              ← Constructed text for embedding + BM25
├── personal: PersonalInfo
│   ├── name: str
│   ├── location: Location (city, state, country, is_remote_ok)
│   ├── languages_spoken: list[str]
│   └── native_language: str | None
├── professional: ProfessionalInfo
│   ├── current_title: str | None
│   ├── current_company: str | None
│   ├── total_experience_years: float | None
│   ├── industry: str | None
│   ├── employment_type: EmploymentType | None
│   └── seniority_level: int | None      ← ★ NEW
├── skills: list[Skill]
│   ├── name: str
│   ├── category: SkillCategory
│   ├── proficiency: ProficiencyLevel | None
│   ├── years_used: float | None
│   ├── evidence: str | None
│   └── confidence: float (0-1)
├── experience: list[WorkExperience]
│   ├── title: str
│   ├── company: str
│   ├── start_date: str | None
│   ├── end_date: str | None
│   ├── is_current: bool
│   ├── description: str
│   ├── highlights: list[str]
│   ├── skills_demonstrated: list[str]
│   └── location: str | None
├── education: list[Education]
├── signals: Signals (is_passive, github_activity, certifications, etc.)
└── metadata: ProfileMetadata (language, quality_score, embedding_vector_id, etc.)
```

**Query/Result chain (lines 151-314):**
```
JobQuery → ParsedQuery → SearchRequest → MatchResult → SearchResponse
                                ↑              ↑
                         SearchFilters    MatchScores + Rationale
```

Key detail: `MatchScores` (line 213) has 9 fields including `overall`, `confidence`, and 7 dimension scores. `SearchResponse` (line 307) wraps everything for API output.

#### `config.py` — Config Loading & LLM Factory

| Function | Line | What it does |
|---|---|---|
| `get_settings()` | 40-42 | Pydantic Settings from env vars + .env file |
| `get_scoring_config()` | 45-47 | Loads `scoring_weights.yaml` (cached) |
| `get_model_config()` | 50-52 | Loads `models.yaml` (cached) |
| `get_app_config()` | 55-57 | Loads `settings.yaml` (cached) |
| `get_llm_client()` | 60-88 | Factory: OpenAI / Gemini / Ollama based on env |

The LLM provider is chosen by `LLM_PROVIDER` env var (openai|gemini|ollama). Change providers without changing code.

#### `profile_store.py` — Lazy Profile Loading

Uses random-access offset index (`offset_index.json`) built by `build_indexes.py`. Profiles are loaded from the JSONL file by byte offset on demand — no database needed.

#### `constants.py` — Indian Market Data

- `INDIAN_LANGUAGES` (list of 12): Hindi, Tamil, Telugu, Marathi, Bengali, etc.
- `INDIAN_COMPANIES` (list of 45): TCS, Infosys, Flipkart, Razorpay, Zoho, etc.
- `INDIAN_CITIES` (list of 20): Bangalore, Mumbai, Delhi, Hyderabad, Pune, Chennai, etc.
- `INDIAN_UNIVERSITIES` (list of 20): IITs, NITs, BITS, IIITs, etc.

---

### 3.2 Extraction (`src/extraction/`) — ★ NEW

This module replaces the blind `.get()` calls in the normalizer with intelligent, rule-based cross-field extraction.

#### `career_history_utils.py` — Shared Helpers

| Function | Line | What it does |
|---|---|---|
| `latest_role()` | 9-15 | Returns current role, or most recent if no current flag |
| `compute_years_from_dates()` | 17-51 | Computes total years from career history date ranges |

**Date malformation handling:**
| Case | Behavior |
|---|---|
| Missing start date | Skip entry entirely |
| Missing end date (not current) | Assume 1 year, mark "low" confidence |
| Missing end date (is_current) | Use today as end |
| End < start | Skip entry |
| End > today | Cap at today |
| Overlapping intervals | Merge (count each day once) |

Returns `(total_years, valid_entry_count, confidence_level)` where confidence is "high", "medium", or "low".

#### `seniority.py` — Seniority Level Inference

Priority order (lines 33-67):
1. Domain override (professor → 6, etc.)
2. Title keyword match (intern → 0, senior → 3, cto → 6)
3. Career history title keyword match
4. Years-based fallback (<2 → junior, <5 → mid, <8 → senior, <12 → lead, else principal+)

**Important:** Years thresholds are tuned for Indian IT market. See `DOMAIN_SPECIFIC_CONSTANTS_NOTE` at line 29.

#### `experience_years.py` — Experience Years Extraction

Priority order (lines 19-47):
1. **Source A:** `years_of_experience` field (with >100 guard)
2. **Source B:** Computed from career history dates
3. **Agreement check:** If both exist and agree within 20% → average them
4. **Disagreement:** Use the higher-confidence source
5. **Regex fallback:** Extract from headline/summary patterns ("8+ yrs experience", "12 years")

Regex is the LAST resort because it's inherently unreliable (line 43-45).

#### `title.py` — Current Title Extraction

Priority order (lines 12-29):
1. `current_title` field (direct)
2. Latest career history title
3. Headline parsing: strip pipe-separated company names, "at Company" suffixes, seniority prefixes

#### `company.py` — Current Company Extraction

Priority order (lines 8-22):
1. `current_company` field (direct)
2. Latest career history company

#### `domain.py` — Industry/Domain Extraction

4 signals (lines 47-73):
1. `current_industry` field (direct)
2. Company name → industry map (30 curated companies: Flipkart→ecommerce, Razorpay→fintech, etc.)
3. Skills → industry cluster (NLP+PyTorch+CV → ai/ml)
4. Headline/summary keywords ("fintech", "healthcare", etc.)

#### `pipeline.py` — Orchestrator

`FieldExtractorPipeline` (lines 24-40):
- `extract(raw_dict)` → `ExtractionBundle` with 5 `ExtractionResult` objects
- Each `ExtractionResult` has `(value, source, confidence)`
- Results are chained: seniority uses title + years results

---

### 3.3 Ingestion (`src/ingestion/`)

#### `parser.py` — ProfileParser

| Method | Line | Format |
|---|---|---|
| `parse_jsonl_file()` | 17 | JSONL (streaming generator) |
| `parse_json_file()` | 42 | JSON array or single object |
| `parse_csv_file()` | 56 | CSV with header mapping |
| `parse_docx()` | 70 | DOCX via zipfile+XML |
| `parse_batch()` | 90 | List of any format |

**JSONL streaming** (line 17-40): Uses a generator with `try/except` per line. Invalid JSON lines are skipped, not failed. Supports gzip detection.

#### `normalizer.py` — normalize_redrob()

The critical function (line 40-159). Now uses `FieldExtractorPipeline` for:

- `current_title`: pipeline result → `.get("current_title")` fallback
- `current_company`: pipeline result → `.get("current_company")` fallback
- `total_experience_years`: pipeline result → `.get("years_of_experience")` fallback
- `industry`: pipeline result → `.get("current_industry")` fallback
- `seniority_level`: pipeline result only (new field)

**Zero-risk design:** If every extractor returns None, behavior is identical to the old code.

Other normalization logic:
- Skills: alias-based skill extraction from raw_text (line 126-139)
- Skill categories: keyword matching (line 162-198)
- Grade parsing: regex extract GPA from free-text ("8.24 CGPA" → 8.24)
- Location: split "Bangalore, Karnataka" into city/state
- Languages: detect native from proficiency flags
- Signals: map `redrob_signals` to `Signals` model

#### `extractor.py` — LLM FieldExtractor

For DOCX parsing fallback. Uses an LLM prompt to extract structured fields from unstructured resume text. **Not used for the main JSONL path** — that's what the extraction module handles.

#### `quality_scorer.py` — Data Quality

`compute_data_quality_score()` (line 19-46):
```
Name present:      +0.10
Title present:     +0.10
Has skills:        +0.15
Has experience:    +0.15
Has education:     +0.10
Has location:      +0.10
raw_text > 200:    +0.10
raw_text > 500:    +0.05
No encoding bugs:  +0.05
Skills have evidence: +0.10
```

Profiles with score < 0.3 are filtered out in `build_indexes.py:64`.

---

### 3.4 Language (`src/language/`)

#### `detector.py` — LanguageDetector

Uses `langdetect.detect()`. Returns `(is_english, language_code, needs_translation)`. All 100K Redrob profiles are in English, so this is a no-op for the main dataset.

#### `translator.py` — TranslationPipeline

Uses `deep-translator` (Google Translate, free, no API key). Supports 15 Indian languages. For the main dataset, all profiles are English so translation is unused.

#### `code_mixed.py` — CodeMixedProcessor

Detects Hinglish (Hindi+English mix) using 3 strategies:
1. Devanagari Unicode character detection
2. 185 Hinglish keyword dictionary ("hai", "nahi", "karo", "achha")
3. Latin vs non-Latin word ratio

Has a transliteration map of 80+ words (kaam→work, baat→talk) and regex-based NER fallback for skills/companies.

#### `multilingual.py` — MultilingualEmbedder

Wrapper around `paraphrase-multilingual-MiniLM-L12-v2`:
- 384-dimension output vectors
- `normalize_embeddings=True` so cosine similarity = dot product
- `embed_batch()` with configurable batch_size (default 64)
- Lazy model loading (loaded on first `.model` access)

---

### 3.5 Search (`src/search/`)

#### `vector_search.py` — FAISS

| Method | Line | What it does |
|---|---|---|
| `build_index()` | 18 | Creates `IndexFlatIP(384)`, adds embeddings |
| `search()` | 23 | Query vector → top_k results with scores |
| `save()` | 35 | Writes FAISS index + JSON id_map |
| `load()` | 45 | Reads both from disk |

`IndexFlatIP` = brute force inner product search. For 100K vectors at 384 dims, this takes ~10ms. For 700M (Redrob scale), you'd need `IndexIVFFlat` or `IndexHNSW`.

#### `bm25_search.py` — BM25

| Method | Line | What it does |
|---|---|---|
| `build_index()` | 18 | Tokenizes documents, builds BM25Okapi |
| `search()` | 23 | Tokenizes query, scores all docs |
| `save()` | 36 | Pickles tokenized corpus + id_map |
| `load()` | 45 | Unpickles and rebuilds index |

Tokenization (line 54): `text.lower().split()` — simple whitespace split. No stemming, no stopword removal. BM25 handles term frequency normalization natively.

#### `hybrid.py` — Hybrid Search

**RRF formula** (line 84):
```
score += 1.0 / (k + rank)    # k = 60 (configurable)
```

Parallel execution of FAISS + BM25 via sequential calls (Python threads aren't used — the models are CPU-bound so threading wouldn't help).

**SearchCache** (lines 13-41): LRU cache (256 entries, 60s TTL). Keyed by MD5 of query string. Results are cached for identical queries.

#### `reranker.py` — Cross-Encoder Reranker

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. Takes `(query, profile_text)` pairs and outputs relevance scores.

**Timeout logic** (lines 42-56): If `timeout_ms > 0`, measures elapsed time. If exceeded, falls back to original RRF scores. Default timeout: 500ms.

**Sigmoid normalization** (line 11-12): Maps raw cross-encoder output (unbounded) to (0,1).

#### `filters.py` — SearchFilter

`passes(profile)` checks:
- City match (if profile city != filter city → drop)
- Experience range (if below min or above max → drop)
- Company include/exclude lists

Applied in `executor.py:260-281` AFTER hybrid search, not before. The PRD mentions "scoped pre-search filtering" but current code does it post-search.

---

### 3.6 Matching (`src/matching/`)

#### `skill_matcher.py` — SkillMatcher

4-strategy matching (line 47-80):
1. **Exact:** `name == candidate_skill.name`
2. **Normalized:** `name.lower() == candidate_skill.name.lower()`
3. **Alias:** 32 curated aliases (k8s→kubernetes, react.js→react, ml→machine learning)
4. **Fuzzy:** `SequenceMatcher().ratio() >= 0.8` for typos

Returns `(overall_score, details_list)` where details has per-skill match info.

#### `experience_matcher.py` — ExperienceMatcher

Year scoring (lines 25-44):
- If candidate years < required min → deficit penalty (linear, capped at 0)
- If candidate years > required max → excess penalty (diminishing returns, capped at 2x)
- If no data → 0.5 (neutral)

Industry scoring (lines 46-53):
- Exact match → 1.0
- No match → 0.3
- No data → 0.5

Combined: `0.7 × years_score + 0.3 × industry_score`

#### `scorer.py` — CandidateScorer

**7-dimension scoring** (line 33-43):
```
semantic_similarity:  25%
keyword_match:        15%
skill_match:          30%       ← highest weight
experience_match:     15%
location_match:        5%
education_match:       5%
cross_encoder_score:   5%
```

**Key behavior** (lines 57-64): If a dimension score is None (no data for that candidate), its weight is redistributed — `total_weight` decreases, remaining dimensions proportionally increase in influence. This prevents missing data from penalizing candidates.

**Slider support** (lines 46-55): When slider_weights are provided (from UI), maps UI-facing slider dimensions (`skill_match`, `experience_match`, `education_match`, `assessment_score`, `behavioral_signals`, `cultural_fit`) to internal dimensions via `DIM_TO_ACTUAL` (line 8-15).

**Confidence** (lines 81-86): `1 - std(scores)`. If all dimensions agree (low variance) → high confidence. If they disagree (high variance) → low confidence.

---

### 3.7 Agents (`src/agents/`)

#### `orchestrator.py` — LangGraph State Machine

**AgentState** (TypedDict, lines 30-41):
```python
{
    "raw_query": str,           # Original user query
    "parsed_query": dict,       # LLM-parsed search params
    "results": list[dict],      # MatchResults from executor
    "evaluations": dict,        # Reflector's critique
    "replan_count": int,        # Current replan iteration
    "max_replans": int,         # Default 3
    "should_continue": bool,
    "search_metadata": dict,
    "total_candidates_searched": int,
    "start_time_ms": int,
    "slider_weights": dict,     # From UI
}
```

**StateGraph** (lines 104-125):
```
Nodes: plan → execute → reflect → generate_rationale
Edges: reflect → (replan → plan | done → generate_rationale → END)
```

**Turbo mode** (lines 150-192): For short queries (≤3 words), skips the agent loop entirely. Uses `_parse_query_text()` (lines 67-89) — a simple keyword extractor that:
1. Looks up Indian cities
2. Removes stop words
3. Extracts technical keywords as skills
4. Returns ParsedQuery directly

#### `executor.py` — ExecutorAgent

**`execute()`** (lines 137-231):

1. `_query_to_search_text()` (line 246-258): Concatenates parsed query fields into a search string
2. `hybrid_search.search()` (line 145): Gets top 100 via FAISS+BM25+RRF
3. Also runs separate FAISS+BM25 to get raw scores (lines 147-156) — needed for semantic_similarity and keyword_match dimensions
4. `_apply_filters()` (lines 260-281): Structural filters
5. Cross-encoder rerank (lines 160-168): Top 20 profiles get reranked
6. Per-candidate scoring (lines 170-229):
   - `_skill_match_score()`: Skill overlap (exact→alias→fuzzy→raw_text)
   - Experience match: min(1.0, years / 10.0) — simple normalization
   - Assemble 7 scores → `scorer.compute_overall()`
   - Build MatchResult

**Skill matching** (lines 54-118): Independent of `SkillMatcher` class — duplicate logic for skill matching exists in executor.py. Both use the same 32 aliases.

#### `planner.py` — PlannerAgent

**`plan()`**: Uses LLM with system prompt from `prompts.py` to extract structured params from NL query. Falls back to same `_parse_query_text()` from orchestrator if LLM fails.

**`replan()`**: Takes previous params + feedback from reflector, adjusts criteria (relaxes location, lowers experience bar, broadens skill aliases).

#### `reflector.py` — ReflectorAgent

**`reflect()`**: Evaluates top 10 results using LLM or heuristic threshold. Returns dict with `should_replan` boolean and `feedback` string.

---

### 3.8 Rationale (`src/rationale/`)

#### `generator.py` — RationaleGenerator

Two modes:
1. **LLM mode**: Generates a human-readable explanation per candidate with summary, strengths, gaps, skill details, experience analysis, recommendation
2. **Template mode** (`_template_rationale()`): Falls back to hardcoded template if LLM unavailable

The template generates a basic rationale from skill match percentages.

#### `validator.py` — RationaleValidator

Checks: summary length > 20 chars, strengths is non-empty, recommendation is valid enum. `validate_batch()` for bulk statistics.

---

### 3.9 Fairness (`src/fairness/`)

#### `bias_detector.py` — BiasDetector

4 dimensions (line references vary per method):
1. **Name bias**: Groups by name's first character, compares average scores between groups
2. **Language bias**: Compares English vs non-English profile scores
3. **Location bias**: Tier-1 cities (Bangalore, Mumbai, Delhi) vs tier-2/3
4. **University bias**: IIT/NIT/BITS vs other institutions

Each returns `(detected: bool, observations: str, details: dict)`.

#### `metrics.py` — Fairness Metrics

- `demographic_parity()`: Compute selection rates per group
- `disparate_impact_ratio()`: Ratio of lowest group rate to highest group rate
- 4/5ths rule: DIR < 0.80 → bias detected

---

### 3.10 API (`src/api/`)

#### Routes

| Route | File:Line | Method | Purpose |
|---|---|---|---|
| `/api/v1/search` | `routes/search.py` | POST | Main search endpoint |
| `/api/v1/profiles/{id}` | `routes/profiles.py` | GET | Get single profile |
| `/api/v1/profiles` | `routes/profiles.py` | GET | Paginated profile list |
| `/api/v1/ingest` | `routes/ingest.py` | POST | Bulk ingest profiles |
| `/api/v1/health` | `routes/health.py` | GET | System status |

#### Middleware

- `RequestLoggingMiddleware` (middleware/logging.py): Logs method, path, status, duration
- `InputValidationMiddleware` (middleware/validation.py): Rejects requests > 10MB

#### `main.py` — App Factory

Lifespan (lines 23-96):
1. Check if FAISS index exists → warn if not
2. Load embedding model (pre-heat cache)
3. Load FAISS index + id_map
4. Load BM25 index
5. Create HybridSearch, CrossEncoderReranker, CandidateScorer
6. Load ProfileStore with offset index
7. Create Planner, Executor, Reflector, Orchestrator
8. Register all with route modules

All model loading happens at startup, not per-request. This takes ~3-5s for initialization.

---

### 3.11 Scripts

#### `scripts/build_indexes.py` — Index Builder

The main offline pipeline (lines 27-123):
1. Open JSONL file
2. For each line: parse JSON → `normalize_redrob()` → `compute_data_quality_score()` → filter < 0.3 → append
3. Build raw_texts and document_texts arrays
4. Generate embeddings via MultilingualEmbedder (batch_size=500)
5. Build FAISS index → save
6. Build BM25 index → save
7. Build offset index → save

Supports `--sample N` for testing on a subset, `--force` to rebuild.

#### `scripts/evaluate.py` — Evaluator

Runs evaluation metrics (precision@k, recall@k, nDCG, MRR) against ground truth data.

---

## 4. Configuration Reference

### `configs/settings.yaml`

```yaml
app:
  name: "india-runs"
  version: "0.1.0"

models:
  embedding:
    name: "paraphrase-multilingual-MiniLM-L12-v2"   # 50+ languages
    dimension: 384
    max_seq_length: 256
    device: "cpu"
  cross_encoder:
    name: "cross-encoder/ms-marco-MiniLM-L-6-v2"    # Reranking model
    max_seq_length: 512
    device: "cpu"
  planner:
    provider: "${LLM_PROVIDER:-openai}"              # Env var with default
    model: "gpt-4o-mini"
    temperature: 0.1

search:
  top_k_hybrid: 50        # Candidates from hybrid search
  top_k_final: 10         # Final output count
  rrf_k: 60               # RRF parameter (lower = more aggressive)
  cross_encoder_timeout_ms: 500  # Skip reranker if slow

agent:
  max_replan_cycles: 3
  min_good_matches_for_pass: 8
```

### `configs/scoring_weights.yaml`

```yaml
scoring_weights:
  semantic_similarity: 0.25    # From FAISS cosine similarity
  keyword_match: 0.15          # From BM25 normalized score
  skill_match: 0.30            # From SkillMatcher → executor._skill_match_score
  experience_match: 0.15       # min(1.0, years / 10)
  location_match: 0.05         # Currently always None in executor
  education_match: 0.05        # Currently always None in executor
  cross_encoder_score: 0.05    # From CrossEncoderReranker

slider_weights:                # UI-facing model (6 dims instead of 7)
  skill_match: 0.30
  experience_match: 0.25
  education_match: 0.15
  assessment_score: 0.15
  behavioral_signals: 0.10
  cultural_fit: 0.05

profiency_scores:
  beginner: 0.25
  intermediate: 0.50
  advanced: 0.75
  expert: 1.00
```

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | LLM provider: openai, gemini, ollama |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/india_runs` | PostgreSQL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis |

---

## 5. API Reference

### POST `/api/v1/search`

**Request:**
```json
{
  "query": "Find a senior DevOps engineer with 5+ years experience in AWS and Kubernetes, based in Bangalore",
  "filters": {
    "location": "Bangalore",
    "min_experience_years": 5,
    "remote_ok": false
  },
  "max_results": 10,
  "include_rationale": true,
  "language": "en"
}
```

**Response:**
```json
{
  "query_id": "uuid",
  "total_candidates_searched": 1000,
  "results": [
    {
      "rank": 1,
      "profile_id": "uuid",
      "name": "Rahul Sharma",
      "current_title": "Senior DevOps Engineer",
      "current_company": "Razorpay",
      "location": "Bangalore",
      "experience_years": 7,
      "scores": {
        "overall": 0.92,
        "semantic_similarity": 0.88,
        "keyword_match": 0.95,
        "skill_match": 0.94,
        "experience_match": 0.90,
        "location_match": 1.0,
        "education_match": null,
        "cross_encoder_score": 0.85,
        "confidence": 0.91
      },
      "matched_skills": ["AWS", "Kubernetes", "Docker", "Terraform"],
      "missing_skills": [],
      "rationale": {
        "summary": "Strong match — Rahul has 7 years of DevOps experience...",
        "strengths": ["Direct experience with AWS (3+ years)", "Bangalore location"],
        "gaps": [],
        "skill_details": [...],
        "experience_analysis": "7 years in DevOps, scaled infrastructure...",
        "recommendation": "strong_match"
      },
      "passive_candidate": false,
      "language_matched": false
    }
  ],
  "message": null,
  "suggestions": [],
  "processing_time_ms": 1847,
  "search_metadata": {
    "methods_used": ["hybrid", "cross_encoder_rerank"],
    "replan_count": 0,
    "total_time_ms": 1847
  }
}
```

### Other Endpoints

| Endpoint | Request | Response |
|---|---|---|
| `GET /api/v1/profiles/{id}` | Path: profile_id | Full Profile object |
| `GET /api/v1/profiles?skip=0&limit=10` | Query: skip, limit | Paginated profile list |
| `POST /api/v1/ingest` | Uploaded file | IngestResponse (total, successful, failed) |
| `GET /api/v1/health` | None | HealthResponse (status, index_size, models) |

---

## 6. Testing Strategy

### Test Structure

```
tests/
├── conftest.py                          # Shared fixtures: sample_profile, embeddings, etc.
├── test_extraction/                     # ★ NEW — 63 tests
│   ├── test_career_history_utils.py     #   latest_role, compute_years, date parsing
│   ├── test_seniority.py                #   Title keywords, years fallback
│   ├── test_experience_years.py         #   3-tier heuristic, typo guards
│   ├── test_title.py                    #   Direct → history → headline
│   ├── test_company.py                  #   Direct → history
│   ├── test_domain.py                   #   4-signal industry inference
│   ├── test_pipeline.py                 #   Orchestrator end-to-end
│   └── test_regression.py              #   "old normalizer invariant" — no data loss
├── test_ingestion/                      # Parser, normalizer, quality
├── test_search/                         # FAISS, BM25, hybrid, filters
├── test_matching/                       # Skill matcher, experience, scorer
├── test_agents/                         # Planner fallback, reflector
├── test_api/                            # Endpoint schema, bias detector
├── test_language/                       # Detector, translator, code-mixed, embedder
├── test_rationale/                      # Generator, validator
└── test_integration/                    # End-to-end, imports
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Just extraction tests
pytest tests/test_extraction/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Single test file
pytest tests/test_extraction/test_experience_years.py -v

# Single test
pytest tests/test_extraction/test_experience_years.py::TestExtractExperienceYears::test_direct_field -v
```

### Fixtures in `conftest.py`

- `sample_profile`: Full Profile object with all fields filled
- `sample_profiles`: List of 5 profiles for index testing
- `sample_embeddings`: Random 384-dim numpy array (5 vectors)
- `sample_query_text`: "Find a senior Python developer with experience in Django and AWS"
- `vector_search_instance`: Empty FAISS index for testing
- `bm25_search_instance`: Empty BM25 index for testing

### Test Patterns Used

1. **Unit tests with isolation:** Each extractor is tested independently with known inputs
2. **Regression invariants:** The "new normalizer never loses data" check (test_regression.py)
3. **State verification:** Assert specific fields on Profile objects after normalization
4. **Edge case coverage:** Empty lists, None values, extreme numbers, malformed dates
5. **mypy strict mode:** All code must type-check

### Regression Test Pattern

```python
# tests/test_extraction/test_regression.py
def test_all_non_none_fields_preserved(self):
    for row in SAMPLE_ROWS:
        profile = new_normalizer(row)
        for field in PROFESSIONAL_FIELDS:
            old_val = row.get("profile", {}).get(OLD_FIELD_MAP[field])
            new_val = getattr(profile.professional, field)
            if old_val is not None:
                assert new_val is not None, (
                    f"{field}: was '{old_val}' but new normalizer returned None"
                )
```

---

## 7. How to Extend

### Add a New Extractor

1. Create `src/extraction/new_field.py`:
```python
def extract_new_field(prof: dict, history: list[dict]) -> tuple[value | None, str]:
    # Tier 1: direct field
    # Tier 2: fallback
    return value, "source_tag"
```

2. Add result to `ExtractionBundle` in `pipeline.py`:
```python
@dataclass
class ExtractionBundle:
    ...
    new_field: ExtractionResult = field(default_factory=...)
```

3. Wire it in `FieldExtractorPipeline.extract()`:
```python
new_val, new_src = extract_new_field(prof, history)
bundle.new_field = ExtractionResult(new_val, new_src)
```

4. Use in `normalizer.py`:
```python
professional.new_field = extracted.new_field.value
```

5. Add model field to `ProfessionalInfo` (or wherever it belongs) in `models.py`

6. Write tests in `tests/test_extraction/test_new_field.py`

### Add a New Score Dimension

1. Add field to `MatchScores` in `models.py`
2. Add weight in `configs/scoring_weights.yaml`
3. Compute score in `executor.py:188-196` (the `scores_dict` assembly)
4. Add mapping in `scorer.py:DIM_TO_ACTUAL` if it should appear in slider UI

### Support a New Profile Source

1. Add enum value to `ProfileSource` in `models.py`
2. Create a new normalizer function (e.g., `normalize_naukri()` in `normalizer.py`)
3. The normalizer maps that source's schema → Profile model
4. Call it from `build_indexes.py` or parser

### Modify Scoring Weights

Edit `configs/scoring_weights.yaml` — no Python code changes needed. Changes take effect on next app restart (`get_scoring_config()` is `@lru_cache`).

---

## 8. Debugging & Performance

### Common Gotchas

| Symptom | Likely Cause | Fix |
|---|---|---|
| No results for query | Indexes not built | Run `python scripts/build_indexes.py --sample 50` |
| `FieldExtractorPipeline` import error | Circular import | Check `pipeline.py` imports — use local imports |
| Low `total_experience_years` surprise | Pipeline prefers computed dates over direct field | Check if career_history has accurate dates |
| Cross-encoder timeout | Model too slow on CPU | Increase `cross_encoder_timeout_ms` in settings.yaml |
| LangGraph hangs | LLM call timeout | Check `OPENAI_API_KEY` or switch to Ollama |
| Profile metadata always `language_detected="en"` | All 100K Redrob profiles are English | Normal behavior |
| `seniority_level` is None | No title, no history, no years | Expected for minimal profiles |
| Biased results against tier-2 cities | `bias_detector.py` location check | Use fairness dashboard to verify |

### Performance Budget

| Operation | Cost | Where |
|---|---|---|
| Regex extraction (per profile) | < 0.1ms | src/extraction/ |
| Date computation (per profile) | < 0.3ms | career_history_utils.py |
| `normalize_redrob()` (per profile) | < 0.5ms | normalizer.py |
| Embedding (per profile) | ~5ms | multilingual.py (CPU) |
| FAISS search (100K vectors) | ~10ms | vector_search.py |
| BM25 search (100K docs) | ~40ms | bm25_search.py |
| Cross-encoder (20 pairs) | ~500ms | reranker.py |
| LLM plan query | ~500ms | planner.py |
| LLM reflect results | ~500ms | reflector.py |
| LLM generate rationale | ~200ms/candidate | generator.py |
| **End-to-end (no LLM)** | **~100ms** | |
| **End-to-end (full agent)** | **~1.5-2s** | |

### Logging

All modules use `logging.getLogger(__name__)`. Set log level in `settings.yaml`:
```yaml
app:
  log_level: "DEBUG"  # or INFO, WARNING, ERROR
```

Or via env var:
```bash
export LOG_LEVEL=DEBUG
```

### Tracing a Request

Best place to add debug prints for a new feature:
- `src/agents/executor.py:145` — Before/after hybrid search
- `src/search/hybrid.py:60-61` — Raw FAISS and BM25 results
- `src/matching/scorer.py:57-64` — Dimension weights and scores
- `src/agents/orchestrator.py:134` — AgentState at each node

---

## 9. PRD-to-Code Mapping

| PRD Section | Status | Where Implemented |
|---|---|---|
| FR-1: Profile Ingestion | ✅ | `src/ingestion/parser.py`, `normalizer.py` |
| FR-2: Multilingual Processing | ✅ | `src/language/detector.py`, `translator.py`, `code_mixed.py`, `multilingual.py` |
| FR-3: Hybrid Search | ✅ | `src/search/hybrid.py` (FAISS+BM25+RRF) |
| FR-4: Cross-Encoder Reranking | ✅ | `src/search/reranker.py` |
| FR-5: Agentic Workflow | ✅ | `src/agents/orchestrator.py` (Plan→Execute→Reflect→Re-plan) |
| FR-6: Rationale Generation | ✅ | `src/rationale/generator.py`, `validator.py` |
| FR-7: Confidence Scoring | ✅ | `src/matching/scorer.py` (7 dims, weights from YAML) |
| FR-8: Listwise Tournament | 🔜 | Not implemented — planned for future |
| FR-9: PII Redaction | 🔜 | Not implemented — planned for future |
| Bias Mitigation | ✅ | `src/fairness/bias_detector.py`, `metrics.py` |
| Scoped pre-search filtering | 🔶 Partial | `src/search/filters.py` runs AFTER search, not before |
| Smart Normalizer | ✅ | `src/extraction/` — newly built |

---

## 10. Glossary

| Term | Meaning | Code Reference |
|---|---|---|
| **BM25** | Keyword search algorithm (TF-IDF but smarter) | `bm25_search.py` |
| **Cross-Encoder** | Model that compares (query, doc) pairs directly | `reranker.py` |
| **DIR** | Disparate Impact Ratio — fairness metric (≥0.80) | `fairness/metrics.py` |
| **FAISS** | Facebook AI vector search library | `vector_search.py` |
| **Hinglish** | Hindi+English code-mixed language | `language/code_mixed.py` |
| **LangGraph** | State machine library for LLM workflows | `agents/orchestrator.py` |
| **RRF** | Reciprocal Rank Fusion — combines rankings | `hybrid.py:76-84` |
| **RRF_k** | RRF parameter (default 60): lower = more aggressive | `configs/settings.yaml` |
| **sentence-transformers** | Text→vector conversion library | `language/multilingual.py` |
| **TinT** | Translate-in-Thought: LLM internally translates | `agents/prompts.py` |
| **RR** | Redrob's 700M+ profile database | `data/profiles/candidates.jsonl` |
| **ExtractionBundle** | Container for extracted field results | `extraction/pipeline.py` |
| **FieldExtractorPipeline** | Smart cross-field extraction orchestrator | `extraction/pipeline.py` |
| **Seniority level** | 0-6 integer (intern→cto) | `extraction/seniority.py` |
| **Embedding** | 384-dim vector representing text meaning | `language/multilingual.py` |
| **MatchedSkills** | Skills found in candidate profile | `executor.py:200-202` |
| **MissingSkills** | Skills required but not found | `executor.py:200-202` |
| **Plackett-Luce** | Listwise tournament ranking (NOT implemented yet) | PRD FR-8 |

---

**Where to start reading code:**

1. `src/extraction/pipeline.py` — See how the new smart normalizer works
2. `src/core/models.py` — All data types
3. `src/agents/executor.py` — The core search pipeline
4. `src/search/hybrid.py` — FAISS+BM25+RRF in 85 lines
5. `src/matching/scorer.py` — The scoring math

**Welcome, and good luck!**
