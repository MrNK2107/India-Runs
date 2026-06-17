# Implementation Plan — Intelligent Candidate Discovery System v2.1

> **Execution blueprint for India Runs Track-1 submission.**
> Organized by **modules** with atomic, executable actions per file/subsystem.
> PRD v2.1 source of truth: 25 sections covering competitive landscape, 5-layer architecture, RLHF feedback loop, simplified scoring UI.
>
> **Total modules: 14 | Primary goal: Precision@10 >= 0.85, DIR >= 0.80, cross-lingual MRR >= 0.75**

---

## Module 1: Ingestion & Parsing

**Purpose:** Parse profiles from multiple sources, normalize to unified schema, compute data quality.

**Files:** `src/ingestion/parser.py`, `src/ingestion/normalizer.py`, `src/ingestion/extractor.py`, `src/ingestion/quality_scorer.py`

### Tasks

- [ ] **M1.1** `src/ingestion/parser.py` — Verify `ProfileParser` handles JSONL (streaming), JSON (array), and gzip formats. Confirm `parse_jsonl_file()` yields profiles without loading 487MB file into memory.
- [ ] **M1.2** `src/ingestion/parser.py` — Add `.docx` parsing fallback using `zipfile + xml.etree` for unicode-safe extraction (handles cp1252 breakage).
- [ ] **M1.3** `src/ingestion/normalizer.py` — Verify `normalize_redrob()` maps all 8 Redrob API fields into Profile model (30 Pydantic fields). Fix any mapping gaps.
- [ ] **M1.4** `src/ingestion/normalizer.py` — Ensure `raw_text` construction matches PRD Section 7.2a exactly:
      `"Name: {name}. Title: {title}. Company: {company}. Summary: {summary}. Skills: {...}. Experience: {...}. Education: {...}. Certifications: {...}. Languages: {...}."`
- [ ] **M1.5** `src/ingestion/normalizer.py` — Verify skill categories auto-inferred via keyword matching (lang/framework/tool/domain).
- [ ] **M1.6** `src/ingestion/quality_scorer.py` — Verify `compute_data_quality_score()` scoring: name +0.10, title +0.10, skills +0.15, experience +0.15, education +0.10, location +0.10, raw_text length thresholds, encoding artifacts check, evidence snippets +0.10.
- [ ] **M1.7** `src/ingestion/extractor.py` — Verify `FieldExtractor` LLM-assisted extraction works as async fallback for unstructured data. Confirm prompt returns structured JSON.
- [ ] **M1.8** `src/ingestion/parser.py` — Add noisy profile skip: if quality_score < 0.3, skip profile and increment `failed_profiles` counter.

**Exit Criteria:**
- [ ] 100K Redrob profiles parse without errors
- [ ] `raw_text` matches PRD spec exactly
- [ ] Quality scores computed for all profiles
- [ ] Noisy profiles skipped with counter

---

## Module 2: Language Processing

**Purpose:** Language detection, translation, multilingual embeddings, code-mixed NLP.

**Files:** `src/language/detector.py`, `src/language/translator.py`, `src/language/multilingual.py`, `src/language/code_mixed.py`

### Tasks

- [ ] **M2.1** `src/language/detector.py` — Verify `LanguageDetector.detect()` returns `{language, is_english, needs_translation}` using `langdetect`. Test on Hindi, Tamil, Telugu text.
- [ ] **M2.2** `src/language/detector.py` — Verify `detect_batch()` processes list of texts efficiently.
- [ ] **M2.3** `src/language/translator.py` — Fix `load_models()`: use `transformers.pipeline("translation", ...)` for primary model. Load fallback M2M100 model correctly. (Bug B4 fix)
- [ ] **M2.4** `src/language/translator.py` — Implement `translate_to_english()` that actually translates non-English text. Set `translation_confidence` score.
- [ ] **M2.5** `src/language/translator.py` — On translation failure, set `translation_fallback: true` in metadata. Return original text as fallback.
- [ ] **M2.6** `src/language/translator.py` — Verify `translate_batch()` processes multiple texts with rate limiting.
- [ ] **M2.7** `src/language/multilingual.py` — Verify `MultilingualEmbedder` loads `paraphrase-multilingual-MiniLM-L12-v2` and produces 384-dim vectors.
- [ ] **M2.8** `src/language/multilingual.py` — Verify `normalize_embeddings=True` so cosine similarity = dot product. Test `cosine_similarity()` between English query and Hindi profile > 0.8.
- [ ] **M2.9** `src/language/multilingual.py` — Verify `embed_batch()` processes multiple texts with progress tracking.
- [ ] **M2.10** `src/language/code_mixed.py` — (New file) Implement `CodeMixedProcessor`:
  - `detect_code_mixed(text) -> bool`: checks for mixed Hindi-English script
  - `extract_entities(text) -> list[Entity]`: use HingBERT/HingRoBERTa for NER on code-mixed text
  - `transliterate_hinglish(text) -> str`: convert between Devanagari and Latin script
- [ ] **M2.11** `src/agents/planner.py` — Add Translate-in-Thought (TinT) prompting: if query is code-mixed, prompt LLM to internally translate + parse search params in one step. No explicit translation call.

**Exit Criteria:**
- [ ] Language detection works for 10+ Indian languages
- [ ] Translation pipeline functional with confidence scores
- [ ] Fallback on translation failure works
- [ ] Cross-lingual embedding similarity > 0.8 for equivalent text
- [ ] Code-mixed Hinglish text parsable via HingBERT NER
- [ ] TinT prompting handles code-mixed queries

---

## Module 3: Search & Retrieval

**Purpose:** Hybrid search (BM25 + FAISS + RRF), cross-encoder reranking, scoped pre-search filters, parallel execution.

**Files:** `src/search/vector_search.py`, `src/search/bm25_search.py`, `src/search/hybrid.py`, `src/search/reranker.py`, `src/search/filters.py`

### Tasks

- [ ] **M3.1** `src/search/vector_search.py` — Verify `build_index()` creates FAISS `IndexFlatIP` (384-dim, inner product on normalized vectors). Confirm save/load with JSON `id_map`.
- [ ] **M3.2** `src/search/vector_search.py` — Verify `search()` returns top-k results with distances. Handle empty index gracefully.
- [ ] **M3.3** `src/search/vector_search.py` — Add `add_embeddings(ids, embeddings)` for incremental index updates without full rebuild. Update both FAISS index and `id_map`.
- [ ] **M3.4** `src/search/vector_search.py` — Add auto-rebuild from stored embeddings if FAISS index is corrupted or missing.
- [ ] **M3.5** `src/search/bm25_search.py` — Verify `build_index()` tokenizes with lowercase + split. Confirm save/load with pickle.
- [ ] **M3.6** `src/search/bm25_search.py` — Verify `search()` returns top-k results with BM25 scores.
- [ ] **M3.7** `src/search/filters.py` — Implement `ScopedRetriever`:
  - `get_candidate_ids(filters) -> list[int]`: returns profile IDs passing structural filters
  - Filters: location (city/country/remote), experience (min/max years), companies (include/exclude)
  - If no filters, return ALL profile IDs
  - This narrows search pool BEFORE vector search runs (solves Vector Search Dilution)
- [ ] **M3.8** `src/search/hybrid.py` — Replace sequential BM25 + FAISS execution with parallel using `concurrent.futures.ThreadPoolExecutor`.
- [ ] **M3.9** `src/search/hybrid.py` — `HybridSearch.search()` returns individual FAISS score, BM25 score, AND combined RRF score per candidate (not just combined). Fix Bug B3.
- [ ] **M3.10** `src/search/hybrid.py` — Verify RRF formula: `RRF_score(d) = Σ 1/(k + rank_i(d))` with k=60 from `scoring_weights.yaml`.
- [ ] **M3.11** `src/search/reranker.py` — Verify `CrossEncoderReranker` loads `cross-encoder/ms-marco-MiniLM-L-6-v2` and scores (query, profile_summary) pairs.
- [ ] **M3.12** `src/search/reranker.py` — Verify timeout fallback: if reranking exceeds `cross_encoder_timeout_ms`, skip and return hybrid results directly.
- [ ] **M3.13** `src/agents/executor.py` — Wire scoped retrieval into execution pipeline:
  - Step 1: Call `ScopedRetriever.get_candidate_ids(filters)` FIRST
  - Step 2: Pass narrowed IDs into `HybridSearch.search()`
  - Step 3: Cross-encoder reranking on narrowed results
  - Step 4: Build MatchResults with per-dimension scores (`semantic_similarity` from FAISS, `keyword_match` from BM25)

**Exit Criteria:**
- [ ] FAISS index builds, searches, saves, loads correctly
- [ ] BM25 index builds, searches, saves, loads correctly
- [ ] Scoped filters narrow pool before vector search
- [ ] BM25 + FAISS run in parallel
- [ ] Individual FAISS/BM25 scores tracked separately
- [ ] Cross-encoder reranks with timeout fallback
- [ ] Incremental FAISS updates work
- [ ] Auto-rebuild on corruption works

---

## Module 4: Matching & Scoring

**Purpose:** Skill matching (fuzzy + semantic), experience scoring, weighted overall score computation, confidence calculation.

**Files:** `src/matching/skill_matcher.py`, `src/matching/experience_matcher.py`, `src/matching/scorer.py`, `src/matching/confidence.py`, `configs/scoring_weights.yaml`

### Tasks

- [ ] **M4.1** `src/matching/skill_matcher.py` — Verify `SkillMatcher.find_best_match()` strategies:
  - Exact match (normalized)
  - Alias match (32 SKILL_ALIASES: "react.js" → "react", "ml" → "machine learning")
  - Fuzzy match (token overlap)
- [ ] **M4.2** `src/matching/skill_matcher.py` — Verify `compute_proficiency_match()` scoring: beginner=0.25, intermediate=0.50, advanced=0.75, expert=1.00.
- [ ] **M4.3** `src/matching/skill_matcher.py` — Verify required skills (weight 1.0) vs preferred (0.6) vs nice-to-have (0.3) weighted scoring.
- [ ] **M4.4** `src/matching/experience_matcher.py` — Verify `compute_experience_match()`:
  - Years: deficit (ratio*0.7) vs excess (diminishing returns capped at 2x)
  - Industry: exact=1.0, semantically similar=0.6, different=0.3
- [ ] **M4.5** `src/matching/scorer.py` — Verify `CandidateScorer.compute()` loads weights from `scoring_weights.yaml` at runtime (never hardcoded).
- [ ] **M4.6** `src/matching/scorer.py` — Verify default scoring formula (PRD FR-7.2):
      `overall = 0.25*semantic + 0.15*keyword + 0.30*skill + 0.15*experience + 0.05*location + 0.05*education + 0.05*cross_encoder`
- [ ] **M4.7** `src/matching/scorer.py` — Verify renormalization when dimensions are null (location not specified, education not specified).
- [ ] **M4.8** `src/matching/scorer.py` — Verify `cross_encoder_score` defaults to 0.5 (neutral) if not reranked.
- [ ] **M4.9** `src/matching/confidence.py` — Verify `compute_confidence()` returns high confidence when all dimensions agree, low when they disagree (based on score variance / std dev).
- [ ] **M4.10** `configs/scoring_weights.yaml` — Verify all weights present: 6 scoring dimensions, skill importance (3 levels), proficiency scores (4 levels), `rrf_k: 60`, fairness thresholds.

**Exit Criteria:**
- [ ] Skill matching works for exact, alias, fuzzy matches
- [ ] Experience scoring handles deficit/excess correctly
- [ ] Weights loaded from YAML at runtime
- [ ] Renormalization works when dimensions are null
- [ ] Confidence computation based on variance

---

## Module 5: Ranking (Plackett-Luce)

**Purpose:** Listwise tournament ranking using Plackett-Luce model. Primary innovation differentiator.

**Files:** `src/ranking/listwise_ranker.py` (new), `configs/scoring_weights.yaml`

### Tasks

- [ ] **M5.1** `src/ranking/listwise_ranker.py` — Implement `PlackettLuceRanker` class:
  - `rank(candidates, anonymized_profiles) -> list[(profile_id, merit_score)]`
  - Divide candidates into random groups of 4-5
  - LLM judges each group simultaneously (listwise, not pairwise)
  - Aggregate partial rankings via Plackett-Luce MM algorithm
  - Max 20 EM iterations
  - 3 tournament rounds with reshuffled groups
  - Fallback to pointwise scoring if LLM fails
- [ ] **M5.2** `src/ranking/listwise_ranker.py` — Implement group judge LLM prompt:
  - Present 4-5 anonymized candidates (only skills, experience years, industry — no PII)
  - Output format: comma-separated list of candidate indices in rank order
  - Structured output parsing with regex fallback
- [ ] **M5.3** `src/ranking/listwise_ranker.py` — Implement Plackett-Luce MM algorithm:
  - Initialize merit parameters theta = 1.0 for all candidates
  - EM iteration: update theta based on partial rankings
  - Sort candidates by final theta value (descending)
- [ ] **M5.4** `configs/scoring_weights.yaml` — Add listwise ranking config:
  ```yaml
  listwise_ranking:
    enabled: true
    group_size: 5
    max_em_iterations: 20
    num_tournament_rounds: 3
  ```
- [ ] **M5.5** `src/agents/orchestrator.py` — Add `listwise_ranking` node in agent state machine after cross-encoder reranking:
  - Takes top-20 from cross-encoder
  - Runs through `PlackettLuceRanker`
  - Produces final ordering
  - Falls back to pointwise ordering if LLM unavailable

**Exit Criteria:**
- [ ] Plackett-Luce ranker produces consistent rankings
- [ ] LLM group judge works with structured output parsing
- [ ] MM algorithm converges within 20 iterations
- [ ] Fallback to pointwise on LLM failure
- [ ] Integrated into orchestrator as final ranking step

---

## Module 6: Agentic Workflow

**Purpose:** LangGraph state machine (Plan → Execute → Reflect → Re-plan). LLM-powered query parsing and candidate evaluation.

**Files:** `src/agents/orchestrator.py`, `src/agents/planner.py`, `src/agents/executor.py`, `src/agents/reflector.py`, `src/agents/prompts.py`

### Tasks

- [ ] **M6.1** `src/agents/prompts.py` — Verify PLANNER prompt extracts structured search params from NL query. Test with: "Find a senior DevOps engineer with 5+ years in AWS and Kubernetes, Bangalore".
- [ ] **M6.2** `src/agents/prompts.py` — Verify REFLECTOR prompt evaluates candidates as strong/good/potential/weak match with specific evidence.
- [ ] **M6.3** `src/agents/prompts.py` — Verify RATIONALE prompt generates detailed report with all 12 dimensions (M7 tasks).
- [ ] **M6.4** `src/agents/prompts.py` — Verify REPLAN prompt broadens criteria when too few good matches found.
- [ ] **M6.5** `src/agents/planner.py` — Verify `PlannerAgent.plan()`:
  - Primary: LLM-based structured extraction
  - Fallback: keyword extraction (skill aliases, regex years, city/company matching)
  - `replan()` relaxes requirements when triggered
- [ ] **M6.6** `src/agents/planner.py` — Add TinT prompt strategy (from M2.11): detect code-mixed queries and prompt LLM to internally translate + parse.
- [ ] **M6.7** `src/agents/executor.py` — Verify `ExecutorAgent.execute()` runs the full search pipeline with scoped retrieval integration (from M3.13):
  - Structural filters → Parallel BM25 + FAISS → RRF fusion → Cross-encoder reranking → Scoring
  - Returns list of `MatchResult` with per-dimension scores
- [ ] **M6.8** `src/agents/reflector.py` — Verify `ReflectorAgent.reflect()`:
  - Primary: LLM-based evaluation per candidate
  - Fallback: score-threshold evaluation
  - `should_replan()` heuristic returns true if < 8/10 candidates are good matches
- [ ] **M6.9** `src/agents/reflector.py` — Wire anonymization: ensure profiles are anonymized before LLM evaluation (uses Anonymizer from M8.1).
- [ ] **M6.10** `src/agents/orchestrator.py` — Verify LangGraph state machine:
  - Nodes: planner, executor, reflector, rationale_generator, listwise_ranker
  - Edges: planner→executor→reflector→(replan→planner OR generate)
  - Max `replan_cycles` from config (default 3)
  - State tracked in `AgentState` TypedDict
- [ ] **M6.11** `src/agents/orchestrator.py` — Fix `_rationale_node()` stub. Wire `RationaleGenerator.generate()` for each result profile. (Bug B2 fix)
- [ ] **M6.12** `src/agents/orchestrator.py` — Add listwise ranking node (from M5.5) between reranking and rationale generation.
- [ ] **M6.13** `src/api/routes/search.py` — Pass all request fields (filters, language, include_rationale) to orchestrator. (Bug B1 fix)

**Exit Criteria:**
- [ ] Planner extracts structured params from NL queries
- [ ] Executor runs full search pipeline with scoped retrieval
- [ ] Reflector evaluates candidates with fallback
- [ ] Re-plan triggers correctly when < 8/10 matches are good
- [ ] Max 3 re-plan cycles respected
- [ ] Rationale generated in agent loop
- [ ] Listwise ranking integrated
- [ ] TinT works for code-mixed queries

---

## Module 7: Rationale Generation

**Purpose:** 12-dimension YAML rationale reports with evidence-based evaluation. PRD Section 14.

**Files:** `src/rationale/generator.py`, `src/rationale/templates.py`, `src/rationale/validator.py`

### Tasks

- [ ] **M7.1** `src/rationale/generator.py` — Expand `RationaleGenerator` to evaluate all 12 dimensions:
  1. `core_technical_skills` — exact match of required programming languages and frameworks
  2. `tool_proficiency` — CI/CD, cloud, monitoring tools
  3. `domain_expertise` — industry-specific knowledge (fintech, ecom, healthcare)
  4. `role_stability` — average tenure, job changes
  5. `leadership_indicators` — team lead, architect, mentoring roles
  6. `communication_signals` — public speaking, technical writing, open source
  7. `multilingual_fit` — language capabilities matching query requirements
  8. `localized_salary_alignment` — compensation parity for location/seniority
  9. `career_growth_trajectory` — promotions, expanding responsibilities
  10. `industry_relevance` — experience in same sector as role
  11. `company_prestige` — brand-name companies vs startups (signal, not bias)
  12. `culture_fit_signals` — open source, side projects, community involvement
- [ ] **M7.2** `src/rationale/generator.py` — Implement YAML output format (PRD Section 14.1):
  ```yaml
  candidate_evaluation:
    profile_id: "..."
    match_confidence: 0.92
    dimensional_scores:
      core_technical_skills: 0.95
      ...
    matching_evidence:
      - dimension: "core_technical_skills"
        skill: "React JS"
        proven_years: 3.5
        context_found: "Lead UI Developer at FinTech Startup"
        proficiency_match: true
    evaluation_rationale:
      summary: "..."
      strengths: [...]
      gaps: [...]
      recommendation: "strong_match"
    anonymization_note: "..."
  ```
- [ ] **M7.3** `src/rationale/generator.py` — Add `anonymization_note` field to all outputs: "PII stripped before LLM evaluation — no name, university, or location data was visible to the evaluator".
- [ ] **M7.4** `src/rationale/generator.py` — Wire anonymized profiles (from M8.x) into generator instead of raw profiles.
- [ ] **M7.5** `src/rationale/templates.py` — Add `RATIONALE_12D_TEMPLATE` that explicitly requests evaluation across all 12 dimensions with evidence for each.
- [ ] **M7.6** `src/rationale/validator.py` — Update `validate()` to check:
  - All 12 dimensions present with valid scores (0.0-1.0)
  - `matching_evidence` has >= 1 entry per matched dimension
  - `evaluation_rationale.summary` not empty
  - `anonymization_note` present
  - Recommendation is one of: strong_match, good_match, potential_match, weak_match
- [ ] **M7.7** `src/rationale/validator.py` — `validate_batch()` computes pass/fail statistics. Verify `should_continue` logic.

**Exit Criteria:**
- [ ] 12 dimensions in rationale output with valid scores
- [ ] YAML output format matches PRD spec
- [ ] Matching evidence references specific profile data
- [ ] Anonymization note present in every report
- [ ] Validator checks all 12 dimensions

---

## Module 8: Fairness & Bias

**Purpose:** PII redaction, style anonymization, bias detection, automated fairness halting. PRD Section 15.

**Files:** `src/fairness/anonymizer.py` (new), `src/fairness/bias_detector.py`, `src/fairness/metrics.py`, `configs/scoring_weights.yaml`

### Tasks

- [ ] **M8.1** `src/fairness/anonymizer.py` — (New file) Implement `Anonymizer` class:
  - `anonymize_profile(profile) -> dict` strips:
    - Name → `"Candidate-{uuid8}"`
    - Gendered pronouns → neutral (they/them)
    - University names → `"University-{tier}"` (tier-1, tier-2, tier-3)
    - Company names → `"Company-{size}-{domain}"`
    - Specific addresses → city only
    - Photo URLs → removed
  - Preserves: skills, experience years, industries, seniority, certifications
- [ ] **M8.2** `src/fairness/anonymizer.py` — Implement `style_anonymize(text) -> str`:
  - Remove excessive bullet-point structures
  - Replace overused LLM verbs: spearheaded, fostered, architected, orchestrated, pioneered, championed, drove, delivered
  - Strip generic LLM power phrases
  - Normalize formatting artifacts
- [ ] **M8.3** `src/agents/executor.py` — Wire `Anonymizer.anonymize_profile()` before passing profiles to reflector and rationale generator.
- [ ] **M8.4** `src/agents/reflector.py` — Use anonymized profiles (not raw) for LLM evaluation.
- [ ] **M8.5** `src/rationale/generator.py` — Use anonymized profiles for rationale generation. Add `anonymization_note`.
- [ ] **M8.6** `src/api/routes/search.py` — Add `anonymization_note` to search response metadata.
- [ ] **M8.7** `src/fairness/bias_detector.py` — Verify 4 bias checks:
  - Name-based (first character grouping)
  - Language-based (en vs non-en scores)
  - Location-based (tier-1 vs tier-2/3 cities)
  - University-based (IIT/NIT/BITS vs other)
  - Each returns: detected flag, observations, details
- [ ] **M8.8** `src/fairness/metrics.py` — Fix `compute_disparate_impact_ratio()`. Rewrite `_in_group` to check protected group against profile attributes independently of education entries. (Bug B5 fix)
- [ ] **M8.9** `src/fairness/metrics.py` — Add `compute_equal_opportunity()`:
  - TPR comparison: correctly matched candidates in group / total relevant in group
  - Returns ratio between protected and majority groups
- [ ] **M8.10** `src/fairness/metrics.py` — Implement automated DIR halting:
  - After every search, compute fairness metrics
  - If DIR < 0.80, add `fairness_warning` to SearchResponse
  - Warning: "Disparate Impact Ratio {value:.2f} < 0.80. Results flagged for review."
  - Flag in UI with warning badge

**Exit Criteria:**
- [ ] PII stripped before LLM evaluation (names, universities, companies removed)
- [ ] Style anonymization removes LLM-writing artifacts
- [ ] Skills/years/industry preserved after anonymization
- [ ] All 4 bias checks functional
- [ ] Equal opportunity metric computed
- [ ] DIR < 0.80 triggers automated flagging

---

## Module 9: Feedback Loop & RLHF (NEW — PRD v2.1)

**Purpose:** Recruiter accept/reject feedback tracking, score re-weighting, RLHF-style retraining. Architecture Principle #10, Layer 5.

**Files:** `src/feedback/tracker.py` (new), `src/feedback/reweighter.py` (new), `src/feedback/store.py` (new), `src/api/routes/feedback.py` (new), `configs/scoring_weights.yaml`

### Tasks

- [ ] **M9.1** `configs/scoring_weights.yaml` — Add feedback loop config:
  ```yaml
  feedback:
    enabled: true
    store_size: 10000
    min_feedback_for_retrain: 50
    reweight_learning_rate: 0.1
  ```
- [ ] **M9.2** `src/feedback/store.py` — (New file) Implement `FeedbackStore`:
  - `store_feedback(query_id, profile_id, action: accept|reject, timestamp)` — append feedback to local JSON store
  - `get_feedback(limit=1000) -> list[FeedbackEntry]` — retrieve recent feedback
  - `get_stats() -> dict` — total accept/reject counts, per-dimension breakdown
  - File-backed persistence (JSON lines in `data/feedback/`)
- [ ] **M9.3** `src/feedback/tracker.py` — (New file) Implement `FeedbackTracker`:
  - `track_match_outcome(match_result, action)` — record recruiter decision for a match
  - `get_per_dimension_accuracy() -> dict` — compute how well each scoring dimension predicted recruiter decisions
  - Dimensions analyzed: skill_match, experience_match, location_match, education_match, semantic_similarity, keyword_match, cross_encoder_score
- [ ] **M9.4** `src/feedback/reweighter.py` — (New file) Implement `ScoringReweighter`:
  - `reweight(feedback_entries) -> dict` — adjust scoring weights based on feedback:
    - Dimensions that correlate with accepted candidates get UP-weighted
    - Dimensions that correlate with rejected candidates get DOWN-weighted
    - Learning rate from config (default 0.1)
    - New weights normalized to sum to 1.0
  - `get_current_weights() -> dict` — return current active weights
  - `reset_weights()` — restore weights from `scoring_weights.yaml` defaults
- [ ] **M9.5** `src/api/routes/feedback.py` — (New file) Implement feedback endpoint:
  - `POST /api/v1/feedback` — accepts `{query_id, profile_id, action: "accept"|"reject"}`
  - Stores via FeedbackStore
  - Triggers FeedbackTracker update
  - If enough feedback accumulated (min_feedback_for_retrain), triggers ScoringReweighter
  - Returns updated feedback stats
- [ ] **M9.6** `src/main.py` — Import and mount feedback router at `/api/v1/feedback`. Initialize FeedbackStore on startup.
- [ ] **M9.7** `src/agents/orchestrator.py` — Add optional feedback node at end of pipeline:
  - After returning results, store processing metadata for potential future feedback
  - No pipeline delay — feedback is async/post-process

**Exit Criteria:**
- [ ] Feedback accepted via API endpoint
- [ ] Feedback persisted to file-backed store
- [ ] Per-dimension accuracy computed
- [ ] Scoring weights adjust based on feedback patterns
- [ ] Retrain triggers after minimum feedback threshold
- [ ] Config-driven enable/disable

---

## Module 10: API Layer

**Purpose:** FastAPI endpoints, middleware, health monitoring.

**Files:** `src/main.py`, `src/api/routes/search.py`, `src/api/routes/profiles.py`, `src/api/routes/ingest.py`, `src/api/routes/health.py`, `src/api/routes/feedback.py`, `src/api/middleware/logging.py`, `src/api/middleware/validation.py`, `src/api/middleware/rate_limit.py`, `src/api/middleware/metrics.py`

### Tasks

- [ ] **M10.1** `src/main.py` — Verify lifespan creates indexes, loads models on startup. Register all middleware.
- [ ] **M10.2** `src/main.py` — Import and register `InputValidationMiddleware`. (Bug B7 fix)
- [ ] **M10.3** `src/main.py` — Import and mount feedback router at `/api/v1/feedback`.
- [ ] **M10.4** `src/api/routes/search.py` — Fix: pass `request.filters`, `request.language`, `request.include_rationale` to orchestrator. (Bug B1 fix)
- [ ] **M10.5** `src/api/routes/search.py` — Add `anonymization_note` to search response metadata.
- [ ] **M10.6** `src/api/routes/search.py` — Handle 0-results: return message + suggestions string array.
- [ ] **M10.7** `src/api/routes/profiles.py` — Verify `GET /api/v1/profiles/{id}` returns full profile. Verify `GET /api/v1/profiles` pagination.
- [ ] **M10.8** `src/api/routes/ingest.py` — Verify `POST /api/v1/ingest` validates filename, parses JSON, returns ingestion report (profiles processed, errors, language distribution).
- [ ] **M10.9** `src/api/routes/health.py` — Fix: add `init_health()` to inject real model loading state. Return index sizes and `last_updated`. (Bug B6 fix)
- [ ] **M10.10** `src/api/middleware/logging.py` — Replace basic INFO logging with structured JSON entries: timestamp, method, path, status_code, latency_ms. PII redacted.
- [ ] **M10.11** `src/api/middleware/validation.py` — Verify InputValidationMiddleware rejects requests > 10MB body.
- [ ] **M10.12** `src/api/middleware/rate_limit.py` — (New file) Implement rate limiting middleware: 429 with `retry-after` header.
- [ ] **M10.13** `src/api/middleware/metrics.py` — (New file) Implement in-memory metrics tracker: request count, error count, latency p50/p95/p99. Exposed via `/api/v1/health`.

**Exit Criteria:**
- [ ] All 4 original endpoints working (search, profiles, ingest, health)
- [ ] Feedback endpoint working
- [ ] Input validation active
- [ ] Rate limiting functional
- [ ] Structured JSON logging active
- [ ] Metrics tracking operational
- [ ] Health endpoint reflects real model state

---

## Module 11: UI (Gradio)

**Purpose:** Demo application with search, results, rationale viewer, scoring slider, analytics, dark mode. PRD Section 16 + FR-7.2a.

**Files:** `src/ui/app.py`, `src/ui/components.py`, `src/ui/styles.css`

### Tasks

- [ ] **M11.1** `src/ui/app.py` — Verify 3-tab layout: Search, Analytics, About. Each tab loads correctly.
- [ ] **M11.2** `src/ui/app.py` — Verify search tab has: large input, example chips, advanced filters, search button.
- [ ] **M11.3** `src/ui/app.py` — Add Hindi example queries (PRD Section 16.1):
  - "Mujhe ek DevOps engineer chiye jisko AWS aur Kubernetes mein 5 saal ka experience hai"
  - "Python aur data science mein 3 saal ka anubhav wala candidate dhoondhein"
- [ ] **M11.4** `src/ui/app.py` — Implement left panel / right panel layout:
  - Left: ranked candidate cards (scrollable list)
  - Right: full rationale on card click
  - Use Gradio's `gr.Row` + `gr.Column` with visibility toggling
- [ ] **M11.5** `src/ui/app.py` — Replace hardcoded analytics with live computed values:
  - Language distribution from actual profiles
  - Source distribution from actual profiles
  - Average match scores from results
  - Passive vs active ratio
  - Fairness metrics from BiasDetector + Metrics module
- [ ] **M11.6** `src/ui/app.py` — Add fairness warning badge when DIR < 0.80.
- [ ] **M11.7** `src/ui/components.py` — Create `create_candidate_card()`: score badge (color-coded), skill chips, quick rationale summary.
- [ ] **M11.8** `src/ui/components.py` — Create `create_score_radar_chart()`: SVG radar chart showing all 12 dimensional scores.
- [ ] **M11.9** `src/ui/components.py` — Create `create_skill_match_table()`: skill, required, found, proficiency_match, evidence columns.
- [ ] **M11.10** `src/ui/components.py` — Create `create_analytics_dashboard()`: CSS grid with metric cards + bar chart.
- [ ] **M11.11** `src/ui/components.py` — Create `create_rationale_panel()`: color-coded summary, strengths (green), gaps (red), recommendation badge.
- [ ] **M11.12** `src/ui/components.py` — Create `create_experience_timeline()`: horizontal timeline with role, company, dates, color-coded by relevance.
- [ ] **M11.13** `src/ui/components.py` — Create `create_loading_spinner()`: animated skeleton placeholder while search runs. Wire into `app.py` search flow.
- [ ] **M11.14** **NEW — Scoring Slider UI (FR-7.2a):** Implement interactive slider for simplified scoring model:
  - 6 sliders: Skill match (30%), Experience (25%), Education (15%), Assessment (15%), Behavioral (10%), Cultural fit (5%)
  - Sliders update in real-time with recalculation of final fit score
  - Dimension mapping from slider model to internal model:
    | Slider Dimension | Internal Mapping |
    |------------------|-----------------|
    | Skill match (30%) | `semantic_similarity + skill_match` weighted blend |
    | Experience (25%) | `experience_match + keyword_match` weighted blend |
    | Education (15%) | `education_match` |
    | Assessment (15%) | `cross_encoder + skill_match` (certifications) |
    | Behavioral (10%) | `keyword_match` (activity signals, engagement) |
    | Cultural fit (5%) | `semantic_similarity` (work style preferences) |
  - Formula display: `score = (s1×0.30) + (s2×0.25) + (s3×0.15) + (s4×0.15) + (s5×0.10) + (s6×0.05)`
  - Visual: color-coded final score (red < 40, yellow 40-60, blue 60-80, green > 80)
- [ ] **M11.15** `src/ui/styles.css` — Add dark mode CSS variables. Add toggle button. Use `prefers-color-scheme` media query for auto-detection.
- [ ] **M11.16** `src/ui/styles.css` — Ensure responsive layout (works on laptop/tablet). Skeleton screen animations.

**Exit Criteria:**
- [ ] 3-tab layout functional
- [ ] Search with Hindi examples works
- [ ] Left/right panel layout with click-to-select
- [ ] Live analytics data from pipeline
- [ ] Interactive scoring slider with real-time recalculation
- [ ] Experience timeline component
- [ ] Dark mode toggle working
- [ ] Skeleton loading on search
- [ ] Fairness warning badge visible

---

## Module 12: Error Handling & Observability

**Purpose:** Graceful degradation for all failure scenarios, structured logging, metrics.

**Files:** `src/agents/orchestrator.py`, `src/language/translator.py`, `src/ingestion/parser.py`, `src/search/vector_search.py`, `src/search/reranker.py`, `src/api/middleware/rate_limit.py`, `src/api/middleware/metrics.py`, `src/api/middleware/logging.py`

### Tasks

- [ ] **M12.1** `src/agents/orchestrator.py` — 0-results: populate `message` + `suggestions` array in SearchResponse when no candidates match.
- [ ] **M12.2** `src/language/translator.py` — Translation failure: set `translation_fallback: true` in metadata, return original text.
- [ ] **M12.3** `src/ingestion/parser.py` — Noisy profile: if quality_score < 0.3, skip profile and increment `failed_profiles` counter.
- [ ] **M12.4** `src/agents/planner.py` — LLM unavailable: fall back to keyword extraction via spaCy NER.
- [ ] **M12.5** `src/rationale/generator.py` — LLM unavailable: return template-based rationale (already implemented, verify).
- [ ] **M12.6** `src/search/vector_search.py` — FAISS index corrupted: auto-rebuild from stored embeddings in PostgreSQL.
- [ ] **M12.7** `src/search/reranker.py` — Cross-encoder too slow: skip reranking if latency > `cross_encoder_timeout_ms`, return hybrid results directly.
- [ ] **M12.8** `src/api/middleware/rate_limit.py` — (New) Rate limit exceeded: return 429 with `retry-after` header.
- [ ] **M12.9** `src/api/middleware/logging.py` — Structured JSON logging: timestamp, method, path, status_code, latency_ms. PII redacted.
- [ ] **M12.10** `src/api/middleware/metrics.py` — (New) In-memory metrics: request count, error count, latency p50/p95/p99. Exposed via health endpoint.

**Exit Criteria:**
- [ ] All 8 fallback scenarios verified
- [ ] Structured JSON logging operational
- [ ] Metrics tracking with p50/p95/p99
- [ ] Rate limiting functional

---

## Module 13: Testing

**Purpose:** Comprehensive test suite: unit, integration, e2e. CI pipeline.

**Files:** All files under `tests/`, `.github/workflows/test.yml`

### Tasks

- [ ] **M13.1** `tests/test_ingestion/test_extractor.py` — (New) LLM field extraction with mocked LLM.
- [ ] **M13.2** `tests/test_ingestion/test_normalizer.py` — (New) Field normalization for all profile sources.
- [ ] **M13.3** `tests/test_language/test_translator.py` — (New) Actual translation + fallback + code-mixed Hinglish via HingBERT.
- [ ] **M13.4** `tests/test_search/test_vector.py` — (New) FAISS build/search/save/load/incremental add/auto-rebuild.
- [ ] **M13.5** `tests/test_search/test_bm25.py` — (New) BM25 build/search/save/load.
- [ ] **M13.6** `tests/test_search/test_reranker.py` — (New) Cross-encoder scoring + timeout fallback.
- [ ] **M13.7** `tests/test_matching/test_scorer.py` — (New) Weighted scoring + YAML loading + renormalization.
- [ ] **M13.8** `tests/test_matching/test_confidence.py` — (New) Confidence calculation + variance.
- [ ] **M13.9** `tests/test_agents/test_orchestrator.py` — (New) Full state machine + replan cycles.
- [ ] **M13.10** `tests/test_agents/test_reflector.py` — (New) LLM + fallback + anonymization integration.
- [ ] **M13.11** `tests/test_api/test_health_endpoint.py` — (New) Health endpoint with real model state.
- [ ] **M13.12** `tests/test_matching/test_listwise_ranker.py` — (New) Plackett-Luce aggregation, LLM judge, fallback.
- [ ] **M13.13** `tests/test_fairness/test_anonymizer.py` — (New) PII stripped, style artifacts removed, skills preserved.
- [ ] **M13.14** `tests/test_fairness/test_metrics.py` — (New) DIR halting, equal opportunity, bias detection.
- [ ] **M13.15** `tests/test_search/test_hybrid.py` — Add critical cases:
  - `test_cross_lingual_search_matches_hindi_to_english`
  - `test_multilingual_embedding_same_space`
  - `test_reranker_improves_precision`
  - `test_scoped_retrieval_filters_before_search`
  - `test_bm25_and_faiss_run_in_parallel`
- [ ] **M13.16** `tests/test_matching/test_skill_matcher.py` — Add critical cases:
  - `test_semantic_skill_match` ("cloud computing" matches "AWS")
  - `test_required_skill_missing_penalizes`
  - `test_nice_to_have_skill_bonuses`
- [ ] **M13.17** `tests/test_integration/test_end_to_end.py` — Add critical cases:
  - `test_full_pipeline_end_to_end`
  - `test_replan_triggered_on_poor_results`
  - `test_max_replan_limit`
  - `test_search_with_multilingual_profiles`
  - `test_search_with_messy_profiles`
  - `test_search_latency_under_2s`
- [ ] **M13.18** `.github/workflows/test.yml` — (New) CI pipeline:
  ```yaml
  name: CI
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with: { python-version: "3.11" }
        - run: pip install -e ".[dev]"
        - run: ruff check src/
        - run: python -m pytest tests/ -v --tb=short
  ```
- [ ] **M13.19** `src/rationale/validator.py` — Add `validate_batch()` for rationale quality statistics. Used in tests.

**Exit Criteria:**
- [ ] 14 new test files created
- [ ] 12+ critical test cases added to existing files
- [ ] CI pipeline running
- [ ] 100+ total tests passing
- [ ] `ruff check src/` clean
- [ ] 80%+ code coverage

---

## Module 14: Documentation & Submission

**Purpose:** README, docs, pitch deck, deployment configuration, final submission.

**Files:** `README.md`, `docs/*.md`, `Dockerfile`, `docker-compose.yml`, `gradio_deploy.py`, `.github/workflows/test.yml`

### Tasks

- [ ] **M14.1** `README.md` — Update with:
  - Problem statement and solution overview
  - Architecture diagram (5-layer + listwise + PII redaction)
  - Key metrics targets (precision@10 >= 0.85, etc.)
  - Quick start with all 3 LLM providers
  - Deployment options table (local, Docker, Spaces, Railway)
  - Tech stack summary
- [ ] **M14.2** `docs/architecture.md` — Update: add listwise ranking, scoped retrieval, RLHF feedback loop, competitive landscape context.
- [ ] **M14.3** `docs/api.md` — Update: add feedback endpoint, update response schemas with new fields (anonymization_note, fairness_warning).
- [ ] **M14.4** `docs/evaluation.md` — Update: add listwise ranking evaluation metrics, feedback loop metrics.
- [ ] **M14.5** `docs/deployment.md` — Update: HuggingFace Spaces config, Railway config, environment variables for all providers.
- [ ] **M14.6** `Dockerfile` — Fix: expose port 7860 for Gradio, run `build_indexes.py` on startup.
- [ ] **M14.7** `docker-compose.yml` — Add Gradio service alongside FastAPI app.
- [ ] **M14.8** `gradio_deploy.py` — (New) HuggingFace Spaces entry point.
- [ ] **M14.9** `docs/pitch_deck.pdf` — (New) 13-slide pitch deck:
  1. Title: "Intelligent Candidate Discovery — Beyond Keywords"
  2. Problem: Keyword matching failure (DevOps example)
  3. Problem: Indian market challenges
  4. Competitive Landscape: Why existing tools fall short
  5. Solution: 5-layer architecture diagram
  6. Innovation: Agentic workflow (Plan → Execute → Reflect → Re-plan)
  7. Innovation: Hybrid search + Explainable rationales
  8. Innovation: Multilingual + PII-free evaluation
  9. Demo: Search + Scoring sliders + Rationale report
  10. Metrics: Precision, Recall, Latency, Cross-lingual MRR
  11. Impact: Passive talent, fairness, accessibility
  12. Roadmap: Path to production
  13. Thank you + Contact

**Exit Criteria:**
- [ ] README updated with all sections
- [ ] All 4 docs/*.md files updated
- [ ] Dockerfile + docker-compose working
- [ ] Gradio Spaces config ready
- [ ] Pitch deck PDF complete
- [ ] GitHub repo public

---

## File Change Summary

| File | Change Type | Module |
|------|-------------|--------|
| `src/ingestion/parser.py` | Verify + noisy skip | M1 |
| `src/ingestion/normalizer.py` | Fix raw_text, verify | M1 |
| `src/ingestion/extractor.py` | Verify | M1 |
| `src/ingestion/quality_scorer.py` | Verify | M1 |
| `src/language/detector.py` | Verify | M2 |
| `src/language/translator.py` | Bug fix + full translation | M2 |
| `src/language/multilingual.py` | Verify | M2 |
| `src/language/code_mixed.py` | **New file** | M2 |
| `src/search/vector_search.py` | Incremental FAISS + auto-rebuild | M3 |
| `src/search/bm25_search.py` | Verify | M3 |
| `src/search/hybrid.py` | Parallel execution + individual scores | M3 |
| `src/search/reranker.py` | Verify + timeout fallback | M3 |
| `src/search/filters.py` | ScopedRetriever | M3 |
| `src/matching/skill_matcher.py` | Verify | M4 |
| `src/matching/experience_matcher.py` | Verify | M4 |
| `src/matching/scorer.py` | Verify YAML loading + renormalization | M4 |
| `src/matching/confidence.py` | Verify | M4 |
| `src/ranking/listwise_ranker.py` | **New file** | M5 |
| `src/agents/prompts.py` | Verify + 12D rationale prompt | M6, M7 |
| `src/agents/planner.py` | TinT prompt | M2, M6 |
| `src/agents/executor.py` | Scoped retrieval + anonymization wiring | M3, M6, M8 |
| `src/agents/reflector.py` | Anonymization integration | M6, M8 |
| `src/agents/orchestrator.py` | Listwise node + feedback node + fix stub | M5, M6, M9 |
| `src/rationale/generator.py` | 12 dimensions + YAML + anonymized profiles | M7, M8 |
| `src/rationale/templates.py` | 12D prompt template | M7 |
| `src/rationale/validator.py` | 12D validation + batch stats | M7 |
| `src/fairness/anonymizer.py` | **New file** | M8 |
| `src/fairness/bias_detector.py` | Verify | M8 |
| `src/fairness/metrics.py` | Bug fix + equal opportunity + DIR halting | M8 |
| `src/feedback/tracker.py` | **New file** | M9 |
| `src/feedback/reweighter.py` | **New file** | M9 |
| `src/feedback/store.py` | **New file** | M9 |
| `src/api/routes/search.py` | Fix filters + anonymization_note + 0-results | M6, M8, M10, M12 |
| `src/api/routes/feedback.py` | **New file** | M9, M10 |
| `src/api/routes/health.py` | Fix hardcoded state | M10 |
| `src/api/middleware/logging.py` | Structured JSON | M10, M12 |
| `src/api/middleware/validation.py` | Register middleware | M10 |
| `src/api/middleware/rate_limit.py` | **New file** | M10, M12 |
| `src/api/middleware/metrics.py` | **New file** | M10, M12 |
| `src/main.py` | Register middleware + feedback router | M9, M10 |
| `src/ui/app.py` | Scoring slider, live analytics, Hindi examples | M11 |
| `src/ui/components.py` | Slider, timeline, 12D chart, skeleton | M11 |
| `src/ui/styles.css` | Dark mode, slider styles | M11 |
| `configs/scoring_weights.yaml` | Add fairness + listwise + feedback configs | M4, M5, M9 |
| `tests/*` (14 new files + updates) | All test files | M13 |
| `.github/workflows/test.yml` | **New file** | M13 |
| `README.md` | Update | M14 |
| `docs/*.md` (4 files) | Update | M14 |
| `docs/pitch_deck.pdf` | **New file** | M14 |
| `Dockerfile` | Fix ports + startup | M14 |
| `docker-compose.yml` | Add Gradio service | M14 |
| `gradio_deploy.py` | **New file** | M14 |
