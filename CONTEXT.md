# Project Context — India Runs

> Last updated: June 17, 2026 by opencode agent
> Lines: 78/300

## Current Status
Pipeline runs end-to-end: server start → load indexes (50 profiles) → POST /api/v1/search → returns 50 ranked candidates with scores/skills/rationale. 3 core pipeline bugs fixed (infinite replan loop, missing rationale field, profile loading). Interactive scoring sliders (6 dimensions) with live re-rank. 86 tests pass, ruff clean, 0 warnings.

## Active Tasks
- [x] Phase A3: Wire UI to return results (src/main.py lifespan)
- [x] Phase B1: Scoring slider UI (6 recruiter sliders, live re-rank)
- [x] Phase B3: Better candidate cards (score bars, color badges)
- [ ] Phase C1: Plackett-Luce listwise ranking
- [ ] Phase C2: PII anonymizer
- [ ] Phase D2: Generate submission CSV

## Architecture Decisions
- **UI Framework:** Gradio only — simpler for demos, free HF Spaces hosting
- **Search:** Hybrid BM25 + FAISS + RRF fusion + cross-encoder reranking
- **Scoring:** 6 recruiter-facing slider dimensions (Skill, Experience, Education, Assessment, Behavioral, Cultural Fit). Sliders map to actual score fields via DIM_TO_ACTUAL. Live re-rank via Gradio state cache.
- **Agents:** LangGraph state machine (Plan → Execute → Reflect → Re-plan). Max 3 replans (increment bug FIXED).
- **LLM:** Multi-provider (OpenAI, Gemini, Ollama). Falls back gracefully when no API key.
- **Translation:** deep-translator (Google Translate, free) — replaces opus-mt models (~300MB each)

## Key Files
| File | Purpose |
|------|---------|
| `configs/scoring_weights.yaml` | Tunable scoring weights (10 dimensions + 6 slider dims) |
| `src/main.py` | FastAPI startup with lifespan: loads indexes, profiles, creates orchestrator |
| `src/agents/orchestrator.py` | LangGraph workflow with replan_count fix |
| `src/agents/executor.py` | Computes skill_match + experience_match from profile data |
| `src/matching/scorer.py` | Multi-dim scoring with slider weight override |
| `src/ui/app.py` | Gradio UI with 6 scoring sliders + live re-rank |
| `src/ui/components.py` | Candidate cards with score breakdown bars + color badges |

## Known Issues
- Pip dependency conflict with supabase packages (httpx<0.28) — unrelated
- Phase 2 (synthetic data) obsoleted by real dataset
- First request slow (~40s) due to lazy model loading. Set HF_TOKEN for faster downloads.
- No real ground truth data — evaluate.py falls back to demo mode with 5 sample queries

## Environment
- HF_TOKEN set via env var — eliminates unauthenticated warnings on model download
- Scoring weights: default Skill=30%, Exp=20%, Education=10%, Assessment=10%, Behavioral=15%, Cultural Fit=10%
- Embedding model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim, 50+ langs)
- Cross-encoder: ms-marco-MiniLM-L6-v2

## Recent History
- Sessions 1-18: PRD, IMPLEMENTATION_PLAN, Phase 0-15 (all 15 phases)
- Session 19: PRD v2.1 — competitive landscape, 5-layer ref
- Session 20: 3 architecture fixes (translation, constants, vector scoring) + build_indexes + evaluate + Phase A3/B1/B3
