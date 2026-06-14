# Project Context — India Runs

> Last updated: June 14, 2026 by current session
> ⚠️ Lines: 54/600 — keep under limit!

## Current Status
Phase 0 complete. Phase 1 (core infrastructure) complete — configs, config.py, constants.py, models.py all written, linted, and import-validated. Ready for Phase 2 (synthetic data generation).

## Active Tasks
- [ ] Phase 2: Synthetic data generation (1,000 profiles, 50 queries)
- [ ] Phase 3: Ingestion pipeline
- [ ] Phase 4: Language pipeline
- [ ] Phase 5: Search pipeline

## Architecture Decisions
- **UI Framework:** Gradio only (not Streamlit) — simpler for demos, free HuggingFace Spaces hosting
- **Embedding Model:** paraphrase-multilingual-MiniLM-L12-v2 (384-dim, 50+ languages, local)
- **Search:** Hybrid BM25 + FAISS + RRF fusion + cross-encoder reranking
- **Agents:** LangGraph state machine (Plan → Execute → Reflect → Re-plan)
- **LLM:** Multi-provider support — OpenAI (GPT-4o-mini), Google Gemini, local Ollama. Configurable via LLM_PROVIDER env var.
- **Scoring weights:** Loaded from configs/scoring_weights.yaml at runtime, never hardcoded

## Key Files
| File | Purpose |
|------|---------|
| `PRD.md` | Complete product requirements document (24 sections) |
| `IMPLEMENTATION_PLAN.md` | File-by-file execution blueprint (78 files, ~7230 lines) |
| `.agent-rules.md` | Mandatory rules for all agents (context, task log, git) |
| `CONTEXT.md` | This file — shared cross-session context |
| `TASK_LOG.md` | Activity log for all agents |
| `configs/settings.yaml` | Application settings with env var interpolation |
| `configs/scoring_weights.yaml` | Tunable scoring weights (6 dimensions) |
| `configs/models.yaml` | ML model configurations (embedding, cross-encoder, translation) |
| `src/core/config.py` | Pydantic Settings, YAML loaders, get_llm_client() factory |
| `src/core/constants.py` | Supported languages, Indian companies/cities/universities, FAISS paths |
| `src/core/models.py` | 30 Pydantic models + 7 StrEnums (Profile, JobQuery, MatchResult, API) |

## Known Issues
- Pip dependency conflict with supabase packages (httpx<0.28) — unrelated, not a project issue

## Environment
- Project path: `C:\Users\nanda\Desktop\india-runs`
- OS: Windows (win32), bash shell
- Python 3.11+ required
- Docker available for PostgreSQL + Redis
- No virtual environment set up yet

## Recent History
- Session 1: Researched hackathon, created PRD, IMPLEMENTATION_PLAN, agent rules, context/task log
- Session 2: Added multi-provider LLM support (OpenAI, Gemini, Ollama) to all docs
- Session 3: Phase 0 implemented — pyproject.toml, Docker, .env.example, configs, directory structure, placeholders
- Session 4: Phase 1 implemented — settings.yaml, scoring_weights.yaml, models.yaml, config.py, constants.py, models.py. ruff clean, all imports validated.
