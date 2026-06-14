# Project Context — India Runs

> Last updated: June 14, 2026 by current session
> ⚠️ Lines: 60/600 — keep under limit!

## Current Status
All 15 phases complete. PRD.md + IMPLEMENTATION_PLAN.md updated to v2 with architectural blueprint strategies (listwise ranking, PII redaction, scoped retrieval, multi-dimensional rationale, code-mixed NLP). All 57 tests pass, ruff clean. Repository pushed to GitHub.

## Architecture Decisions
- **UI Framework:** Gradio only (not Streamlit) — simpler for demos, free HuggingFace Spaces hosting
- **Embedding Model:** paraphrase-multilingual-MiniLM-L12-v2 (384-dim, 50+ languages, local)
- **Search:** Hybrid BM25 + FAISS + RRF fusion + cross-encoder reranking
- **Agents:** LangGraph state machine (Plan → Execute → Reflect → Re-plan)
- **LLM:** Multi-provider support — OpenAI (GPT-4o-mini), Google Gemini, local Ollama
- **Scoring weights:** Loaded from configs/scoring_weights.yaml at runtime, never hardcoded
- **TranslationPipeline is stubbed:** opus-mt models too large for 16GB/5min CPU constraint; all 100K profiles are English

## Key Files
| File | Purpose |
|------|---------|
| `PRD.md` | Complete product requirements document (24 sections) |
| `IMPLEMENTATION_PLAN.md` | File-by-file execution blueprint (78 files, ~7230 lines) |
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
