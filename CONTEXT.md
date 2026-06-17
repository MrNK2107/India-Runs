# Project Context — India Runs

> Last updated: June 17, 2026 by opencode agent
> ⚠️ Lines: 56/300 — keep under limit!

## Current Status
All 15 phases complete. Three core architectural issues fixed: (1) Translation replaced heavy opus-mt models with deep-translator (Google Translate, free, no API key), (2) Constants expanded from ~85 to 400+ entries (cities 20→120, universities 20→60, companies 45→120+), (3) Vector embeddings now properly used for semantic similarity scoring (previously both semantic_similarity and keyword_match used the same RRF rank position score). 86 tests pass, ruff clean.

## Architecture Decisions
- **UI Framework:** Gradio only (not Streamlit) — simpler for demos, free HuggingFace Spaces hosting
- **Embedding Model:** paraphrase-multilingual-MiniLM-L12-v2 (384-dim, 50+ languages, local)
- **Search:** Hybrid BM25 + FAISS + RRF fusion + cross-encoder reranking
- **Scoring:** Semantic_similarity uses actual FAISS cosine similarity (normalized to 0-1), keyword_match uses normalized BM25 raw score — no longer both using same RRF rank
- **Agents:** LangGraph state machine (Plan → Execute → Reflect → Re-plan)
- **LLM:** Multi-provider support — OpenAI (GPT-4o-mini), Google Gemini, local Ollama
- **Scoring weights:** Loaded from configs/scoring_weights.yaml at runtime, never hardcoded
- **Translation:** deep-translator / Google Translate API (free, no API key needed) — replaces opus-mt models (~300MB each)

## Key Files
| File | Purpose |
|------|---------|
| `PRD.md` | Complete product requirements document (25 sections) — v2.1 with competitive landscape, 5-layer ref, simplified scoring model |
| `IMPLEMENTATION_PLAN.md` | Module-based execution blueprint (14 modules, atomic action-level tasks) — v2.1 with Feedback/RLHF module, Scoring Slider UI, code-mixed NLP |
| `README.md` | Project overview and quick start |
| `docs/architecture.md` | System architecture deep-dive |
| `docs/api.md` | API documentation with curl examples |
| `docs/evaluation.md` | Evaluation metrics and interpretation |
| `docs/deployment.md` | Deployment guide (local, Docker, Spaces, Railway) |
| `configs/settings.yaml` | Application settings with env var interpolation |
| `configs/scoring_weights.yaml` | Tunable scoring weights (6 dimensions) |
| `configs/models.yaml` | ML model configurations |
| `src/core/config.py` | Pydantic Settings, YAML loaders, get_llm_client() factory |
| `src/core/models.py` | 30 Pydantic models + 7 StrEnums |

## Known Issues
- Pip dependency conflict with supabase packages (httpx<0.28) — unrelated, not a project issue
- Phase 2 (synthetic data generation) obsoleted by real dataset
- ~~TranslationPipeline stubbed with heavy opus-mt models~~ **FIXED**: now uses deep-translator (Google Translate, free)
- ~~Constants too limited for Indian market coverage~~ **FIXED**: 120+ cities, 60+ universities, 120+ companies
- ~~Vector embeddings not used in scoring (both semantic_similarity and keyword_match used same RRF rank)~~ **FIXED**: semantic_similarity now uses actual FAISS cosine similarity, keyword_match uses normalized BM25 score

## Environment
- Project path: `C:\Users\nanda\Desktop\india-runs`
- OS: Windows (win32), bash shell
- Python 3.11+ required
- Docker available for PostgreSQL + Redis
- No virtual environment set up yet

## Recent History
- Sessions 1-2: PRD, IMPLEMENTATION_PLAN, agent rules
- Session 3: Phase 0 (environment setup)
- Session 4: Phase 1 (core/config)
- Session 5: Phase 3 (ingestion)
- Session 6: Phase 4 (language)
- Session 7: Phase 5 (search)
- Session 8: Phase 6 (matching)
- Session 9: Phase 7 (agents)
- Session 10: Phase 8 (rationale)
- Session 11: Phase 9 (fairness)
- Session 12: Phase 10 (API)
- Session 13: Phase 11 (UI)
- Session 14: Phase 12 (index building)
- Session 15: Phase 13 (evaluation)
- Session 16: Phase 14 (testing) + test fixes — 57/57 passing, ruff clean
- Session 17: Phase 15 (documentation) — README, docs/*.md
- Session 18: Final commit + push, PRD v2 + implementation plan v2
- Session 19: PRD v2.1 — added competitive landscape, 5-layer ref, simplified scoring model, RLHF feedback, expanded tech stack from claude.pdf analysis
