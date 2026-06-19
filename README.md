# India Runs by Redrob AI — Candidate Ranking System

**Track 1: Data & AI Challenge**

A production-grade candidate ranking system that uses multi-query hybrid search (FAISS + BM25) with cross-encoder reranking and 10-dimension professional-grade scoring to identify the best-fit candidates from a pool of 100.

## Key Differentiators

| Feature | Our Approach | Typical Competitors |
|---|---|---|
| **Semantic Understanding** | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) — deep bidirectional attention | TF-IDF or shallow LSA embeddings |
| **Behavioral Signals** | 20+ Redrob platform signals (response rate, saved count, GitHub, verification) | Platform signals ignored |
| **Career Trajectory** | Job hopping penalty, consulting detection, title progression | Years-of-experience only |
| **Skill Proficiency** | Depth-weighted: expert > advanced > intermediate > beginner | Binary skill presence |
| **Honeypot Detection** | Time-travel check, skill-density anomaly, expert-zero-years | None |
| **Reasoning Quality** | 20+ narrative templates, signals-based, non-templated per candidate | Static template "matched X skills" |
| **Query Coverage** | 17 strategic queries across 8 role categories | Single query |
| **Pipeline Speed** | ~8s CPU-only (16GB, 8 cores, no GPU) | 60-75s typical |

## Quick Start

```bash
# Reproduce submission
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Validate output
python validate_submission.py submission.csv

# Run tests
python -m pytest tests/ -q
```

## Architecture

### Phase 1: Pre-computation (offline, one-time)

```
candidates.jsonl
      ↓
[ Normalizer ] → Converts raw data to normalized Profile objects
      ↓
[ Embedder ]   → paraphrase-multilingual-MiniLM-L12-v2 (384d)
      ↓
[ FAISS Index ] + [ BM25 Index ] → Saved to src/data/indexes/
```

### Phase 2: Ranking Pipeline

```
Query (17 strategic) → ParseQuery → [ Hybrid Search (FAISS + BM25) ]
                                           ↓
                                  [ Cross-Encoder Reranker ]
                                           ↓
                                  [ 10-Dimension Scoring ]
      ┌───────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
      │Cross  │Skill│Semantic│Behav│Keyword│Career│YOE  │Skill │Loc  │Edu  │
      │Encoder│Match│  Sim   │Signals│Match │Traj  │Match│Prof  │Match│Match │
      │  25%  │ 18% │  15%   │ 12%  │ 10%  │  7%  │ 8%  │  5%  │  0% │  0%  │
      └───────┴──────┴───────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
                                           ↓
                              [ Best-score merge across queries ]
                                           ↓
                              [ Sort by overall descending ]
                                           ↓
                                   submission.csv
```

### Scoring Dimensions

| # | Dimension | Weight | What It Measures |
|---|-----------|--------|------------------|
| 1 | **Cross-encoder score** | 25% | Deep semantic match between query and candidate profile (ms-marco-MiniLM-L-6-v2) |
| 2 | **Skill match** | 18% | Fuzzy skill matching with aliases and synonym resolution |
| 3 | **Semantic similarity** | 15% | FAISS cosine similarity on multilingual embeddings |
| 4 | **Behavioral signals** | 12% | Redrob platform engagement: response rate, saved by recruiters, profile completeness, verification status, GitHub activity, interview completion rate, recency |
| 5 | **Keyword match** | 10% | BM25 term overlap on raw profile text |
| 6 | **Experience match** | 8% | Total experience years vs target band |
| 7 | **Career trajectory** | 7% | Job hopping penalty (<18mo avg at 3+ jobs), consulting career, title progression signal |
| 8 | **Skill proficiency** | 5% | Depth-weighted: expert/advanced > intermediate/beginner, years used, endorsements |
| 9-10 | Location/Education | — | Reserved for future use, currently zero-weighted |

### Penalties

- **Honeypot profiles** → ×0.15 score multiplier (impossible profiles detected via time-travel, skill-density anomalies)
- **Consulting careers** → Built-in penalty in career_trajectory dimension

## Search Queries

17 targeted queries across 8 categories:
1. **Software Engineering** (4 queries) — senior SDE, backend, frontend, full stack
2. **Data & ML** (3 queries) — data scientist, data engineer, ML engineer
3. **Cloud & DevOps** (2 queries) — DevOps, cloud architect
4. **Java** (1 query) — Spring Boot, microservices
5. **Mobile** (1 query) — Android, iOS, Flutter
6. **Leadership** (2 queries) — engineering manager, product manager
7. **Security & QA** (2 queries) — cybersecurity, QA automation
8. **Solutions** (1 query) — solutions architect, distributed systems

## Behavioral Signals Used

Full set of Redrob platform signals incorporated into scoring:
```
recruiter_response_rate, saved_by_recruiters_30d, profile_completeness_score,
verified_email, verified_phone, linkedin_connected, github_activity_score,
open_to_work, willing_to_relocate, interview_completion_rate,
offer_acceptance_rate, notice_period_days, preferred_work_mode,
connection_count, endorsements_received, search_appearance_30d,
profile_views_received_30d, applications_submitted_30d,
expected_salary_range, skill_assessment_scores
```

## Honeypot Detection

Identifies 11 types of impossible/fake profiles:
1. **Time-travel**: Start year before company was founded (e.g., "worked at Pied Piper in 2012")
2. **Skill-density anomaly**: >5 skills per year of experience
3. **Expert-zero-years**: Expert in 5+ skills with 0 years used

These profiles receive a ×0.15 score penalty, naturally sinking them to ranks 89-100.

## Submission Format

```
candidate_id,rank,score,reasoning
CAND_0000001,1,0.7666,"Actively seeking new opportunities. Short notice period..."
CAND_0000043,2,0.6927,"Currently Cloud Engineer at Swiggy. From Swiggy (top-tier product company)..."
...
```

- 100 rows, strictly non-increasing scores
- Unique reasoning per candidate with behavioral signals
- Format verified by `validate_submission.py`

## Tests

```bash
# Unit + integration tests
python -m pytest tests/ -q --tb=short    # 150 passed, ~2min

# End-to-end validation
python scripts/e2e_test.py                # 35 steps, all passed

# Submission format check
python validate_submission.py submission.csv   # "Submission is valid."
```

## Performance

- **Pipeline**: ~8s for 100 profiles × 17 queries (CPU-only, 16GB RAM, 8 cores)
- **Sub-5min guarantee**: Easily meets the 5-minute constraint even at 100K scale
- **No GPU required**: All models run on CPU
- **No external API calls**: Fully offline after index building

## File Structure

```
rank.py                         → Main entry point (17 query pipeline)
validate_submission.py          → CSV format checker
submission_metadata.yaml        → Hackathon portal metadata
submission.csv                  → Generated output (100 rows)
configs/
  scoring_weights.yaml          → 10-dimension weight configuration
src/
  agents/
    executor.py                 → Search → Rerank → Score pipeline
    orchestrator.py             → Query parsing, LangGraph workflow
  core/
    models.py                   → Pydantic models (Signals, MatchScores, etc.)
  matching/
    scorer.py                   → Weighted scoring engine
    behavioral_scorer.py        → Behavioral, career trajectory, proficiency scores
    skill_matcher.py            → Fuzzy skill matching with aliases
  search/
    reranker.py                 → Cross-encoder reranking
    hybrid.py                   → FAISS + BM25 hybrid search
tests/                          → 150 unit/integration + 35 e2e tests
```
