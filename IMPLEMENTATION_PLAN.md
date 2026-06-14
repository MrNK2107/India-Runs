# Implementation Plan — Intelligent Candidate Discovery System v2.0
# Bug Fixes → Data → Differentiators → Polish

> **This document is the execution blueprint for completing the India Runs Track-1 submission.**
> It accounts for: the existing codebase (57 tests, all 15 phases done), the 7 critical bugs found in the gap analysis, and the new winning strategies from the Architectural Blueprint.
>
> **Total phases: 10 | Estimated effort: 33 days | Primary goal: Precision@10 >= 0.85, DIR >= 0.80, cross-lingual MRR >= 0.75**

---

## Phase 0: Bug Fixes & API Wiring (Days 1-3)

**Mission:** Fix all 7 critical bugs identified in the gap analysis. Get the core pipeline working correctly before adding any new features.

### Bug B1: API search route ignores filters (src/api/routes/search.py)

**Current:** Line 27 passes request.query only, ignores request.filters, request.language, request.include_rationale.

**Fix:** Pass all request fields to orchestrator.run():
- src/api/routes/search.py: update search_candidates to pass request.filters, request.language, request.include_rationale
- src/agents/orchestrator.py: update run() signature to accept and wire filters into ExecutorAgent.execute()

### Bug B2: Rationale node is stub (src/agents/orchestrator.py:138-139)

**Current:** _rationale_node() returns {"should_continue": False} - no rationale generation in the agent loop.

**Fix:** Wire RationaleGenerator into the agent state machine, calling generator.generate() for each result profile.

### Bug B3: Same score for semantic_similarity and keyword_match (src/agents/executor.py:67-71)

**Current:** Both dimensions get the SAME hybrid fusion score.

**Fix:** Modify HybridSearch.search() to return individual FAISS and BM25 scores alongside the RRF score. Update executor to track separate semantic_similarity and keyword_match values.

### Bug B4: Translation model loading broken (src/language/translator.py:19-25)

**Current:** Tokenizer assigned to _primary, model assigned to _fallback - wrong wiring.

**Fix:** Use transformers.pipeline("translation", ...) for primary. Load fallback model separately.

### Bug B5: compute_disparate_impact_ratio flawed (src/fairness/metrics.py:89-102)

**Current:** _in_group iterates education entries, flagging profiles incorrectly.

**Fix:** Rewrite _in_group to check protected group against profile attributes (language, location, university) independently of education entries.

### Bug B6: Health endpoint hardcoded (src/api/routes/health.py)

**Current:** models_loaded always {"embedding": False, "cross_encoder": False}.

**Fix:** Add init_health() function to inject real model loading state. Return index sizes and last_updated.

### Bug B7: InputValidationMiddleware not registered (src/main.py:35)

**Fix:** Import and register InputValidationMiddleware alongside RequestLoggingMiddleware.

### Exit Criteria for Phase 0
- [ ] All 7 bugs fixed and verified
- [ ] API filters wired into orchestrator
- [ ] Rationale generated in agent loop
- [ ] Individual BM25/FAISS scores tracked
- [ ] Translation models load correctly
- [ ] DIR computation corrected
- [ ] Health endpoint reflects real state
- [ ] InputValidationMiddleware active

---

## Phase 1: Data Generation (Days 4-6)

**Mission:** Build the synthetic dataset. Without this, nothing else works.

### 1.1 scripts/generate_data.py (Full Rewrite)

Replace the current 11-line stub with a full synthetic profile generator producing:
- 1,000 profiles across 5 domains (software engineering, data science, product, design, marketing)
- 20% non-English names (Hindi, Tamil, Telugu)
- 3-7 work experiences per profile
- Raw_text per PRD Section 6.2a spec
- Realistic Indian companies, cities, skills, and universities
- 70% passive candidates, varying data quality scores

### 1.2 data/queries/queries.json

Generate 50 search queries: 15 technical, 15 business, 10 creative, 10 cross-functional.
Each query includes raw_query, parsed structure, and language field.

### 1.3 data/ground_truth/ground_truth.json

Ground truth relevance labels for 20 of the 50 queries. Each query maps to 10 top-relevant profile IDs based on skill/experience/location overlap.

### 1.4 Fix raw_text Construction (src/ingestion/normalizer.py)

Update normalizer to match PRD Section 6.2a:
"Name: {name}. Title: {title}. Company: {company}. Summary: {summary}. Skills: {...}. Experience: {...}. Education: {...}. Certifications: {...}. Languages: {...}."

### 1.5 Build Indexes

Run scripts/build_indexes.py against the generated dataset to create FAISS + BM25 indexes.

### Exit Criteria for Phase 1
- [ ] 1,000 synthetic profiles generated
- [ ] 50 queries generated
- [ ] Ground truth for 20 queries
- [ ] raw_text matches PRD spec
- [ ] FAISS + BM25 indexes built

---

## Phase 2: Infrastructure Fixes (Days 7-10)

**Mission:** Add error handling fallbacks, structured logging, metrics tracking.

### 2.1 Error Handling - All 7 Fallback Scenarios

| Scenario | Location | Implementation |
|----------|----------|----------------|
| 0 results | src/agents/orchestrator.py | Populate message + suggestions in SearchResponse |
| Translation fails | src/language/translator.py | Set translation_fallback: true in metadata |
| Noisy profile skip | src/ingestion/parser.py | If quality_score < 0.3, skip + increment counter |
| LLM unavailable (planner) | src/agents/planner.py | Already done (keyword fallback) |
| LLM unavailable (rationale) | src/rationale/generator.py | Already done (template fallback) |
| FAISS corrupted | src/search/vector_search.py | Auto-rebuild from stored embeddings |
| Rate limit | src/api/middleware/rate_limit.py | New middleware returning 429 with retry-after |

### 2.2 Structured JSON Logging (src/api/middleware/logging.py)

Replace basic INFO logging with structured JSON entries:
- Timestamp, method, path, status_code, latency_ms
- PII redacted from log messages

### 2.3 Metrics Tracking (src/api/middleware/metrics.py)

In-memory metrics tracker:
- Request count, error count, latency p50/p95/p99
- Exposed via /api/v1/health endpoint

### 2.4 No-Results Response

When search returns 0 results, populate message and suggestions in SearchResponse.
Suggestions include: try removing location filter, reduce min experience, move required to preferred.

### Exit Criteria for Phase 2
- [ ] All 7 error scenarios handled
- [ ] Structured JSON logging active
- [ ] Metrics tracking operational
- [ ] 0-results response has suggestions

---

## Phase 3: Scoped Retrieval + Parallel Search (Days 11-13)

**Mission:** Solve Bottleneck A (Vector Search Dilution). Apply filters BEFORE search. Run BM25 + FAISS in parallel.

### 3.1 Scoped Pre-Search Filters (src/search/filters.py)

Create ScopedRetriever class:
- get_candidate_ids(filters) returns profile IDs that pass structural filters
- Filters applied: location, experience years, company inclusion/exclusion
- If no filters, returns ALL profile IDs
- This narrows the search pool BEFORE vector search runs

### 3.2 Parallel BM25 + FAISS (src/search/hybrid.py)

Replace sequential execution with parallel using concurrent.futures.ThreadPoolExecutor + asyncio.gather:
- BM25 and FAISS searches run concurrently
- HybridSearch.search() returns individual FAISS and BM25 scores per candidate (not just combined RRF)

### 3.3 Integrate into Executor (src/agents/executor.py)

- Step 1: Call ScopedRetriever.get_candidate_ids(filters) FIRST
- Step 2: Pass narrowed candidate_ids into HybridSearch.search()
- Step 3: Cross-encoder reranking on narrowed results
- Step 4: Build MatchResults with per-dimension scores (semantic_similarity from FAISS, keyword_match from BM25)

### 3.4 Incremental FAISS Support (src/search/vector_search.py)

Add method to add individual profile embeddings to existing index without full rebuild:
- vector_search.add_embeddings(ids, embeddings)
- Updates both faiss index and id_map

### Exit Criteria for Phase 3
- [ ] Pre-search filters narrow candidate pool
- [ ] BM25 + FAISS run in parallel
- [ ] Individual FAISS/BM25 scores tracked
- [ ] Incremental FAISS updates work

---

## Phase 4: Listwise Tournament Ranking (Days 14-17)

**Mission:** Implement Strategy 1 (Plackett-Luce) from the Blueprint. The primary innovation differentiator.

### 4.1 src/ranking/listwise_ranker.py (New File)

Implement PlackettLuceRanker class:

- rank(candidates, anonymized_profiles) -> list of (profile_id, merit_score) sorted by merit
- Candidates divided into random groups of 4-5
- LLM judges each group simultaneously (listwise, not pairwise)
- Partial rankings aggregated via Plackett-Luce MM algorithm (max 20 EM iterations)
- 3 tournament rounds with reshuffled groups
- Fallback to pointwise scoring if LLM fails

### 4.2 Group Judge Prompt

LLM prompt that presents 4-5 anonymized candidates and asks for relative ordering:
- Only skills, experience years, industry shown (no PII)
- Output format: comma-separated list of candidate numbers in rank order
- Structured output parsing with fallback

### 4.3 Integrate into Orchestrator (src/agents/orchestrator.py)

Add listwise_ranking node (or integrate into _rationale_node):
- Takes top-20 from cross-encoder reranking
- Runs through PlackettLuceRanker
- Produces final ordering

### 4.4 Tests (tests/test_matching/test_listwise_ranker.py)

- test_plackett_luce_aggregation: verify MM algorithm produces consistent results
- test_listwise_ranked_better_than_pointwise: compare precision@10
- test_group_judge_prompt: verify LLM prompt structure
- test_fallback_on_llm_failure: verify pointwise fallback

### Exit Criteria for Phase 4
- [ ] Plackett-Luce ranker functional
- [ ] LLM group judge working with structured output
- [ ] Integrated into orchestrator
- [ ] Tests passing

---

## Phase 5: PII Redaction + Bias Automation (Days 18-21)

**Mission:** Implement PII redaction layer (Strategy 3) and automated fairness halting.

### 5.1 src/fairness/anonymizer.py (New File)

Implement Anonymizer class:

- anonymize_profile(profile) -> dict with PII stripped:
  - Name -> "Candidate-{uuid}"
  - Gendered pronouns -> neutral
  - University names -> "University-{tier}"
  - Company names -> "Company-{size}-{domain}"
  - Specific addresses -> city only
  - Photo URLs removed

- style_anonymize(text) -> str with LLM-writing artifacts removed:
  - Remove excessive bullet-point structures
  - Replace overused verbs: spearheaded, fostered, architected, orchestrated, pioneered, championed
  - Strip generic LLM power phrases
  - Other style normalization

### 5.2 Wire Anonymization into Pipeline

- src/agents/executor.py: anonymize profiles before reflection
- src/agents/reflector.py: use anonymized profiles for LLM evaluation
- src/rationale/generator.py: use anonymized profiles for rationale generation
- src/api/routes/search.py: add anonymization_note to response

### 5.3 Equal Opportunity Metric (src/fairness/metrics.py)

Add compute_equal_opportunity():
- True positive rate comparison between protected and majority groups
- TPR = correctly matched candidates in group / total relevant in group
- Returns ratio between groups

### 5.4 Automated DIR Halting

- After every search, compute fairness metrics
- If DIR < 0.80, add fairness_warning to SearchResponse
- Warning: "Disparate Impact Ratio {value:.2f} < 0.80. Results flagged for review."
- Flag in UI with warning badge

### 5.5 Update scoring_weights.yaml

Add fairness thresholds:
`yaml
fairness:
  disparate_impact_threshold: 0.80
  auto_flag_on_violation: true
`

### 5.6 Tests (tests/test_fairness/)

- test_pii_stripped_before_llm: verify names/universities removed
- test_style_anonymization_removes_llm_artifacts: verify style stripping
- test_dir_below_threshold_flags_results: verify halting logic
- test_equal_opportunity_metric_computed: verify metric
- test_anonymization_preserves_skills: verify skills/years/industry intact

### Exit Criteria for Phase 5
- [ ] PII stripped before LLM evaluation
- [ ] Style anonymization active
- [ ] Equal opportunity metric computed
- [ ] Automated DIR halting with flags
- [ ] Tests passing

---

## Phase 6: 12-20 Dimension Rationale Reports (Days 22-24)

**Mission:** Implement Strategy 2 - Multi-dimensional YAML rationale reports.

### 6.1 Expand Rationale Generator (src/rationale/generator.py)

Update RationaleGenerator to evaluate 12 dimensions:
1. core_technical_skills
2. tool_proficiency
3. domain_expertise
4. role_stability
5. leadership_indicators
6. communication_signals
7. multilingual_fit
8. localized_salary_alignment
9. career_growth_trajectory
10. industry_relevance
11. company_prestige
12. culture_fit_signals

### 6.2 YAML Output Format

Generate YAML rationale reports with:
- dimensional_scores: all 12 dimensions with 0-1 scores
- matching_evidence: per dimension, specific evidence from profile
- evaluation_rationale: summary, strengths, gaps, recommendation
- anonymization_note: PII stripping confirmation

### 6.3 Update Rationale Prompt (src/rationale/templates.py)

New RATIONALE_12D_TEMPLATE that explicitly requests evaluation across all 12 dimensions with evidence for each.

### 6.4 Update RationaleValidator (src/rationale/validator.py)

Validate all 12 dimensions are present with valid scores (0-1). Check matching_evidence has at least one entry per matched dimension. Validate anonymization_note is present.

### 6.5 UI Components (src/ui/components.py)

Add 12-dim radar chart component. Add YAML report viewer tab. Replace old 6-dim chart with 12-dim chart.

### Exit Criteria for Phase 6
- [ ] 12 dimensions in rationale output
- [ ] YAML output format working
- [ ] Validator checks all dimensions
- [ ] UI shows 12-dim radar chart

---

## Phase 7: Code-Mixed NLP + Translation (Days 25-27)

**Mission:** Implement Bottleneck B mitigation - handle Hindi-English code-mixed text.

### 7.1 src/language/code_mixed.py (New File)

Implement CodeMixedProcessor:
- Load HingBERT or HingRoBERTa for NER on code-mixed text
- extract_entities(text) -> list of (entity_text, entity_type, confidence)
- detect_code_mixed(text) -> bool
- transliterate_hinglish(text) -> str (Devanagari <-> Latin)

### 7.2 Translate-in-Thought (TinT) for Planner (src/agents/planner.py)

Add TinT prompt strategy to PlannerAgent:
- Detect if query is code-mixed (Hindi + English)
- If yes, prompt LLM to translate internally + parse search params
- No explicit translation call - LLM does both steps in one response
- TinT prompt: "This query is in Hinglish (Hindi-English mixed). Parse it as if it were English for search purposes."

### 7.3 Fix Translation Pipeline (src/language/translator.py)

Fix load_models:
- Load primary model via transformers.pipeline("translation", ...)
- Load fallback M2M100 model correctly
- translate_to_english() now actually translates non-English text
- Set translation_confidence score
- On failure, set translation_fallback: true in metadata

### 7.4 Tests (tests/test_language/test_translator.py)

- test_hingbert_ner_extracts_skills_from_hinglish
- test_translate_in_thought_handles_code_mixed_query
- test_translation_actual_translates_hindi
- test_translation_fallback_on_failure

### Exit Criteria for Phase 7
- [ ] Code-mixed Hinglish text parsable
- [ ] HingBERT NER extracts skills from code-mixed text
- [ ] TinT prompting for planner works
- [ ] Translation pipeline actually translates
- [ ] Tests passing

---

## Phase 8: UI Polish (Days 28-30)

**Mission:** Demo-ready UI: dark mode, skeleton loading, live analytics data, experience timeline.

### 8.1 Dark Mode (src/ui/styles.css, src/ui/app.py)

Add dark mode CSS variables. Add toggle button in Gradio UI. Use prefers-color-scheme media query for auto-detection.

### 8.2 Skeleton Loading States (src/ui/components.py)

Add create_skeleton_card() component that shows animated placeholder while search runs. Wire into app.py search flow.

### 8.3 Live Analytics Data (src/ui/app.py)

Replace hardcoded fairness metrics values with actual computed values from orchestrator:
- Language distribution from profiles
- Source distribution
- Average match scores from results
- Passive vs active ratio
- Fairness metrics from BiasDetector + Metrics module

### 8.4 Left Panel / Right Panel Layout (src/ui/app.py)

Implement click-to-select behavior:
- Left panel: ranked candidate cards
- Right panel: full rationale on card click
- Use Gradio's gr.Row + gr.Column with visibility toggling

### 8.5 Experience Timeline (src/ui/components.py)

Add create_experience_timeline() component:
- Visual horizontal timeline with company logos (or initials)
- Role, company, dates, description
- Color-coded by relevance to query

### 8.6 Example Queries in Hindi

Add Hindi example queries to search bar:
- "Mujhe ek DevOps engineer chiye jisko AWS aur Kubernetes mein 5 saal ka experience hai"
- "Python aur data science mein 3 saal ka anubhav wala candidate dhoondhein"

### Exit Criteria for Phase 8
- [ ] Dark mode toggle working
- [ ] Skeleton loading on search
- [ ] Live analytics data from pipeline
- [ ] Click-to-select candidate cards
- [ ] Experience timeline component
- [ ] Hindi example queries

---

## Phase 9: Full Test Coverage (Days 31-33)

**Mission:** All missing test files written. CI pipeline set up.

### 9.1 Missing Test Files

Create 11 missing test files from PRD spec:
- tests/test_ingestion/test_extractor.py - LLM field extraction
- tests/test_ingestion/test_normalizer.py - field normalization
- tests/test_language/test_translator.py - actual translation
- tests/test_search/test_vector.py - FAISS build/search/save/load
- tests/test_search/test_bm25.py - BM25 build/search/save/load
- tests/test_search/test_reranker.py - cross-encoder reranking
- tests/test_matching/test_scorer.py - weighted scoring
- tests/test_matching/test_confidence.py - confidence calculation
- tests/test_agents/test_orchestrator.py - full state machine
- tests/test_agents/test_reflector.py - LLM + fallback reflection
- tests/test_api/test_health_endpoint.py - health endpoint

### 9.2 Missing Critical Test Cases

From PRD Section 20.2:
- test_cross_lingual_search_matches_hindi_to_english
- test_multilingual_embedding_same_space
- test_reranker_improves_precision
- test_semantic_skill_match
- test_required_skill_missing_penalizes
- test_nice_to_have_skill_bonuses
- test_full_pipeline_end_to_end
- test_replan_triggered_on_poor_results
- test_max_replan_limit
- test_search_with_multilingual_profiles
- test_search_with_messy_profiles
- test_search_latency_under_2s

### 9.3 CI Pipeline (.github/workflows/test.yml)

`yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check src/
      - run: python -m pytest tests/ -v --tb=short
`

### 9.4 Target

- 100+ total tests (up from 57)
- 80%+ code coverage
- All tests passing in CI
- Ruff clean

### Exit Criteria for Phase 9
- [ ] 11 missing test files created
- [ ] 10+ missing critical test cases added
- [ ] CI pipeline running
- [ ] 100+ tests passing
- [ ] Ruff clean

---

## Phase 10: Documentation + Submission Prep (Days 32-33 overlaps with 9)

**Mission:** Complete documentation, pitch deck, deploy.

### 10.1 Update README.md

- Add new architecture diagram (3-stage + listwise + PII redaction)
- Add new metrics targets
- Add setup instructions for all 3 LLM providers
- Add deployment options table

### 10.2 Update docs/

- docs/architecture.md: add listwise ranking section, scoped retrieval
- docs/api.md: update response schemas with new fields
- docs/evaluation.md: add listwise evaluation metrics
- docs/deployment.md: add Spaces deployment, Railway config

### 10.3 Pitch Deck (docs/pitch_deck.pdf)

12 slides from PRD Section 22.2:
1. Title
2-3. Problem (keyword failure + Indian market)
4-5. Solution architecture (3-stage + listwise + PII)
6. Innovation: Listwise Tournament Ranking
7. Innovation: PII Redaction + Fairness
8. Demo screenshots
9. Metrics
10. Impact
11. Roadmap
12. Thank you

### 10.4 Docker / Deployment Fixes

- Dockerfile: expose port 7860 for Gradio, run build_indexes.py on startup
- docker-compose.yml: add gradio service alongside app
- Add gradio_deploy.py for HuggingFace Spaces
- Add railway.json for Railway deployment

### 10.5 Final Commit + Push

- git add -A
- git commit -m "feat: PRD v2 + Implementation Plan v2 + Blueprint strategies"
- git push origin main

### Exit Criteria for Phase 10
- [ ] README and docs updated
- [ ] Pitch deck PDF ready
- [ ] Dockerfile fixed, exposed ports
- [ ] Deployed to Spaces or Railway
- [ ] GitHub repo public with final commit

---

## Appendix: File Change Summary

| File | Change Type | Phase |
|------|-------------|-------|
| src/api/routes/search.py | Bug fix | 0 |
| src/agents/orchestrator.py | Bug fix + new nodes | 0, 4 |
| src/agents/executor.py | Bug fix + scoped retrieval | 0, 3 |
| src/agents/reflector.py | Anonymization integration | 5 |
| src/agents/planner.py | TinT prompt | 7 |
| src/language/translator.py | Bug fix + full translation | 0, 7 |
| src/language/code_mixed.py | New file | 7 |
| src/fairness/metrics.py | Bug fix + equal opportunity | 0, 5 |
| src/fairness/anonymizer.py | New file | 5 |
| src/search/hybrid.py | Parallel + individual scores | 3 |
| src/search/filters.py | Scoped retriever | 3 |
| src/search/vector_search.py | Incremental FAISS | 3 |
| src/ranking/listwise_ranker.py | New file | 4 |
| src/rationale/generator.py | 12 dimensions + YAML | 6 |
| src/rationale/templates.py | 12D prompt | 6 |
| src/rationale/validator.py | 12D validation | 6 |
| src/ingestion/normalizer.py | Fix raw_text | 1 |
| src/api/middleware/rate_limit.py | New file | 2 |
| src/api/middleware/metrics.py | New file | 2 |
| src/api/middleware/logging.py | Structured JSON | 2 |
| src/main.py | Add middleware | 0 |
| src/api/routes/health.py | Fix hardcoded state | 0 |
| src/ui/app.py | Live analytics, dark mode, layout | 8 |
| src/ui/components.py | Skeleton, timeline, 12D chart | 8 |
| src/ui/styles.css | Dark mode CSS | 8 |
| scripts/generate_data.py | Full rewrite | 1 |
| scripts/build_indexes.py | Minor update | 1 |
| scripts/init_db.py | New file | 2 |
| data/queries/queries.json | New file | 1 |
| data/ground_truth/ground_truth.json | New file | 1 |
| configs/scoring_weights.yaml | Add fairness section | 5 |
| tests/test_matching/test_listwise_ranker.py | New file | 4 |
| tests/test_fairness/ | New files | 5 |
| tests/* (11 missing files) | New files | 9 |
| .github/workflows/test.yml | New file | 9 |
| Dockerfile | Fix exposed ports | 10 |
| docker-compose.yml | Add gradio service | 10 |
| gradio_deploy.py | New file | 10 |
| README.md | Update | 10 |
| docs/*.md | Update | 10 |
| docs/pitch_deck.pdf | New file | 10 |
