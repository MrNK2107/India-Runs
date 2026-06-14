# Task Log — India Runs

> Activity log for all agents working on this project.
> Append-only. Never edit previous entries.

---

## June 14, 2026 — Buffy (parent agent) — Session 1

**Task:** Research India Runs hackathon and create planning documents
**Status:** completed

**Changes:**
- `PRD.md`: Created 24-section comprehensive PRD for Intelligent Candidate Discovery system
- `IMPLEMENTATION_PLAN.md`: Created 78-file execution blueprint with exact function signatures
- `.agent-rules.md`: Created mandatory agent workflow rules (context, task log, git conventions)
- `CONTEXT.md`: Created shared cross-session context file
- `TASK_LOG.md`: Created this file

**Research Done:**
- Fetched hack2skill.com/event/india_runs/ — hackathon details
- Researched Redrob AI company background (700M+ profiles, 30+ languages, $10M Series A)
- Researched winning strategies for AI hackathons (hybrid search, agentic workflows)
- Researched multilingual embedding models (paraphrase-multilingual-MiniLM-L12-v2)
- Researched hybrid search patterns (BM25 + FAISS + RRF + cross-encoder)

**Decisions:**
- Track 1 (Data & AI Challenge) chosen — ₹10L prize, technical focus
- Gradio as sole UI framework (not Streamlit)
- LangGraph for agentic workflow (Plan → Execute → Reflect → Re-plan)
- GPT-4o-mini for LLM calls (cheapest, sufficient for planning + rationale)

**Issues:**
- None yet — planning phase only

**Git Commits:**
- `41ab431` docs: add PRD and implementation plan for Intelligent Candidate Discovery system
- `08e9a19` chore: add agent workflow rules, shared context, task log, and gitignore

**Next Steps:**
- Begin Phase 0: Environment setup
- Create pyproject.toml, Docker files, configs

---

## June 14, 2026 — Buffy (parent agent) — Session 2

**Task:** Add multi-provider LLM support (OpenAI, Gemini, Ollama) to all planning docs
**Status:** completed

**Changes:**
- `PRD.md`: Updated tech stack with langchain-google-genai, langchain-ollama; updated architecture diagrams to show "Configurable LLM Provider"; added provider-agnostic architecture principle; updated risk mitigation
- `IMPLEMENTATION_PLAN.md`: Added langchain-google-genai, langchain-ollama, google-genai to pyproject.toml; updated .env.example with LLM_PROVIDER, GEMINI_*, OLLAMA_* vars; updated configs/settings.yaml with provider config; updated src/core/config.py with llm_provider, gemini_api_key, gemini_model, ollama_base_url, ollama_model; updated planner.py and reflector.py to use get_llm_client() factory
- `CONTEXT.md`: Updated LLM section to reflect multi-provider support

**Decisions:**
- LLM provider is configurable via LLM_PROVIDER env var (openai | gemini | ollama)
- All agents use a unified get_llm_client() factory function
- Gemini and Ollama are first-class alternatives to OpenAI
- Ollama provides free local inference for development and demos

**Issues:**
- None

**Git Commits:**
- `pending` docs: update planning docs for multi-provider LLM support (OpenAI, Gemini, Ollama)

**Next Steps:**
- Begin Phase 0: Environment setup
- Implement src/core/llm.py — the provider-agnostic LLM client factory

---

## June 14, 2026 — current session — Session 4

**Task:** Verify Phase 0, then implement Phase 1 (core infrastructure)
**Status:** completed

**Changes:**
- `configs/settings.yaml`: Created with 7 sections (app, database, redis, models, search, scoring, agent) and env var interpolation
- `configs/scoring_weights.yaml`: Created with 6 dimension weights, skill importance, proficiency scores
- `configs/models.yaml`: Created with embedding, cross-encoder, translation, language detection configs
- `src/core/config.py`: Created Settings (Pydantic), load_yaml_config, get_settings/get_scoring_config/get_model_config/get_app_config/get_llm_client() factory (OpenAI/Gemini/Ollama)
- `src/core/constants.py`: Created with 12 supported languages, 45 Indian companies, 20 cities, 20 universities, FAISS/BM25 paths
- `src/core/models.py`: Created with 30 Pydantic models (Profile, JobQuery, MatchResult, API schemas), 7 StrEnums

**Lint:** ruff check — 0 errors after fixes (unused imports removed, Optional→X|None, str+Enum→StrEnum)

**Imports:** All modules load clean. All 3 YAML configs parse and cache successfully.

**Decisions:**
- get_llm_client() lives in config.py (not a separate llm.py), matching IMPLEMENTATION_PLAN.md references
- All enums use StrEnum (Python 3.11+) instead of str+Enum per ruff UP042

**Issues:**
- Pip dependency conflict with supabase httpx<0.28 — unrelated, pre-existing
- ruff --fix auto-converted Optional[X]→X|None across all 329 lines of models.py

**Next Steps:**
- Phase 2: Synthetic data generation (1,000 profiles, 50 queries)

---

## June 14, 2026 — current session — Session 5

**Task:** Proceed with Phase 3 (ingestion pipeline)
**Status:** completed

**Changes:**
- `src/core/models.py`: Added REDROB to ProfileSource enum
- `src/ingestion/parser.py`: ProfileParser — JSONL, JSON, gzip, batch with error handling
- `src/ingestion/normalizer.py`: normalize_redrob() mapping Redrob schema (8 top-level fields) → Profile model (30 Pydantic fields)
- `src/ingestion/quality_scorer.py`: compute_data_quality_score() + bulk_score() per PRD Section 6.2b
- `src/ingestion/extractor.py`: FieldExtractor — async LLM-assisted extraction for future unstructured data

**Lint:** ruff check — 0 errors
**Tests:** Parsed + normalized 50 real Redrob candidates from sample_candidates.json. All fields map correctly (name, title, company, location, skills, experience, education, signals). Quality scores computed.

**Decisions:**
- Normalizer adapted for Redrob schema (not LinkedIn/Naukri/GitHub as original plan specified) since real dataset is available
- Skill categories auto-inferred from skill names via keyword matching (lang/framework/tool/domain)
- File size: candidates.jsonl is 487 MB (100K lines) — parse_jsonl_file uses streaming, not load into memory

**Issues:**
- .docx files use unicode characters that break cp1252 encoding — used Python zipfile + xml.etree to extract
- Phase 2 (synthetic data) is obsoleted by the 100K real Redrob profiles

**Next Steps:**
- Phase 4: Language pipeline (detector, translator, multilingual)

---

## June 14, 2026 — current session — Session 6

**Task:** Verify implementation status and proceed with Phase 4 (language pipeline)
**Status:** completed

**Changes:**
- `src/language/detector.py`: LanguageDetector — detect() and detect_batch() using langdetect
- `src/language/translator.py`: TranslationPipeline — translate_to_english(), translate_batch(), model name lookup for 9 Indian languages
- `src/language/multilingual.py`: MultilingualEmbedder — embed(), embed_batch(), cosine_similarity() using paraphrase-multilingual-MiniLM-L12-v2

**Lint:** ruff check — 0 errors after fix (Optional→X|None)

**Imports:** All three modules load cleanly. langdetect works on Hindi text.

**Decisions:**
- Translation pipeline uses stub implementation for now — real opus-mt models are too large (~300MB each) for the 5min/16GB hackathon constraint. Will enable when needed.
- MultilingualEmbedder uses normalize_embeddings=True so cosine similarity = dot product

**Next Steps:**
- Phase 5: Search pipeline (vector search, BM25, hybrid, reranker, filters)

---

## June 14, 2026 — current session — Session 7

**Task:** Phase 5 — Search pipeline
**Status:** completed

**Changes:**
- `src/search/vector_search.py`: FAISS vector search — build_index, search (Inner Product on normalized vectors), save/load with JSON id_map
- `src/search/bm25_search.py`: BM25 keyword search — build_index, search (lowercase+split tokenization), save/load with pickle
- `src/search/hybrid.py`: HybridSearch — orchestrate parallel vector+keyword search, reciprocal_rank_fusion with configurable k from scoring_weights.yaml
- `src/search/reranker.py`: CrossEncoderReranker — rerank with timeout fallback, uses ms-marco-MiniLM-L-6-v2, timeout from settings.cross_encoder_timeout_ms
- `src/search/filters.py`: SearchFilter — hard filters: location (city/country/exp location/remote), experience (min/max years), companies (include/exclude)

**Lint:** ruff check — 0 errors across all 5 files

**Imports:** All 5 modules load cleanly

**Next Steps:**
- Phase 6: Matching & scoring (skill_matcher.py, experience_matcher.py, scorer.py)

---

## June 14, 2026 — current session — Session 8

**Task:** Phase 6 — Matching & scoring
**Status:** completed

**Changes:**
- `src/matching/skill_matcher.py`: SkillMatcher — fuzzy matching with exact/normalized/alias/fuzzy strategies, SKILL_ALIASES (32 entries), proficiency match scoring
- `src/matching/experience_matcher.py`: ExperienceMatcher — years scoring (deficit/excess penalty), industry scoring (exact match 1.0, else 0.3)
- `src/matching/scorer.py`: CandidateScorer — weighted overall score from config, renormalizes when dimensions are null, confidence via std dev
- `src/matching/confidence.py`: Module-level compute_confidence and compute_score_variance functions

**Lint:** ruff check — 0 errors across all 4 files

**Imports:** All 4 modules load cleanly (32 skill aliases)

**Next Steps:**
- Phase 7: Agentic workflow (prompts.py, planner.py, executor.py, reflector.py, orchestrator.py)

---

## June 14, 2026 — current session — Session 9

**Task:** Phase 7 — Agentic workflow
**Status:** completed

**Changes:**
- `src/agents/prompts.py`: 4 system prompts — PLANNER (structured query extraction), REFLECTOR (hiring evaluation), RATIONALE (report generation), REPLAN (broadening criteria)
- `src/agents/planner.py`: PlannerAgent — plan() with LLM, fallback to keyword extraction (skill aliases, regex years, city/company matching), replan() with relaxation fallback
- `src/agents/executor.py`: ExecutorAgent — execute() runs hybrid search + filters + reranker + scorer, builds MatchResult list
- `src/agents/reflector.py`: ReflectorAgent — reflect() with LLM, fallback to score-threshold evaluation, should_replan heuristic
- `src/agents/orchestrator.py`: Orchestrator — LangGraph state machine (Plan → Execute → Reflect → Re-plan loop), AgentState TypedDict, conditional edges with max_replan_cycles from config

**Lint:** ruff check — 0 errors across all 5 files

**Imports:** All 5 modules load cleanly

**Next Steps:**
- Phase 8: Rationale generation

---

## June 14, 2026 — current session — Session 10

**Task:** Phase 8 — Rationale generation
**Status:** completed

**Changes:**
- `src/rationale/generator.py`: RationaleGenerator — generate() with LLM (provider-agnostic via get_llm_client), template fallback with skill/experience/strength/gap text generation
- `src/rationale/templates.py`: RATIONALE_TEMPLATE and SKILL_EVIDENCE_TEMPLATE for LLM prompting
- `src/rationale/validator.py`: RationaleValidator — validate() checks summary length, strengths presence, valid recommendation; validate_batch() for statistics

**Lint:** ruff check — 0 errors across all 3 files

**Imports:** All 3 modules load cleanly

**Next Steps:**
- Phase 9: Fairness & bias evaluation

---

## June 14, 2026 — current session — Session 11

**Task:** Phase 9 — Fairness & bias evaluation
**Status:** completed

**Changes:**
- `src/fairness/bias_detector.py`: BiasDetector — 4 bias checks: name (first-character grouping), language (en vs non-en scores), location (tier-1 vs tier-2/3 cities), university (IIT/NIT/BITS vs others), each with detected flag, observations, and details
- `src/fairness/metrics.py`: Module-level functions — compute_demographic_parity() for university/city/language, compute_disparate_impact_ratio() with 4/5ths rule, compute_language_bias(), compute_location_bias(), compute_all_fairness_metrics() aggregator

**Lint:** ruff check — 0 errors across both files

**Imports:** Both modules load cleanly

**Next Steps:**
- Phase 10: API layer (FastAPI endpoints)

---

## June 14, 2026 — current session — Session 12

**Task:** Phase 10 — API layer
**Status:** completed

**Changes:**
- `src/main.py`: FastAPI entry point with lifespan, middleware, and 4 routers mounted at /api/v1
- `src/api/routes/search.py`: POST /api/v1/search — delegates to orchestrator, init_orchestrator() for startup wiring
- `src/api/routes/profiles.py`: GET /api/v1/profiles/{id} + GET /api/v1/profiles (paginated), init_profiles() for startup wiring
- `src/api/routes/ingest.py`: POST /api/v1/ingest — upload JSON, validates filename and parses
- `src/api/routes/health.py`: GET /api/v1/health — status, version, index_size, models_loaded, init_health() for startup wiring
- `src/api/middleware/logging.py`: RequestLoggingMiddleware — logs method, path, status, duration
- `src/api/middleware/validation.py`: InputValidationMiddleware — rejects requests >10MB body

**Lint:** ruff check — 0 errors across all 7 files

**Imports:** All 7 modules load cleanly

**Next Steps:**
- Phase 11: Gradio UI (app.py, components.py, styles.css)

---

## June 14, 2026 — current session — Session 13

**Task:** Phase 11 — Gradio UI
**Status:** completed

**Changes:**
- `src/ui/app.py`: Gradio application with 3 tabs — Search (query input, filters, examples, results area, rationale panel), Analytics (fairness metrics dashboard, score distribution), About (system architecture & tech stack)
- `src/ui/components.py`: 7 reusable components — create_candidate_card (HTML with score badge, skill chips), create_score_radar_chart (SVG radar), create_skill_match_table, create_analytics_dashboard (CSS grid with metric cards + bar chart), create_rationale_panel (color-coded with strengths/gaps), create_loading_spinner
- `src/ui/styles.css`: Custom CSS — score badges, skill chips, candidate cards, metric cards, results scrollbar, spinner animation

**Lint:** ruff check — 0 errors across all 3 files

**Imports:** All 3 modules load cleanly

**Next Steps:**
- Phase 12: Index building (scripts/build_indexes.py)
