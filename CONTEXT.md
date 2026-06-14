# Project Context — India Runs

> Last updated: June 14, 2026 by current session
> ⚠️ Lines: 60/600 — keep under limit!

## Current Status
Phase 0 complete. Phase 1 complete. Phase 3 complete. Phase 4 complete. Phase 5 (search pipeline) complete. Phase 6 (matching & scoring) complete. Phase 7 (agentic workflow) complete. Phase 8 (rationale generation) complete. Phase 9 (fairness & bias) complete. Phase 10 (API layer) complete — FastAPI app with /search, /profiles, /ingest, /health endpoints, request logging & validation middleware. Phase 11 (Gradio UI) complete — search tab, analytics dashboard, about page. Phase 12 (index building) complete — scripts/build_indexes.py loads profiles, generates embeddings, builds FAISS + BM25 indexes. Phase 13 (evaluation) complete — scripts/evaluate.py with P@k, R@k, MRR, nDCG, cross-lingual MRR, latency stats.

## Active Tasks
- [ ] Phase 7: Agentic workflow

## Architecture Decisions
- **UI Framework:** Gradio only (not Streamlit) — simpler for demos, free HuggingFace Spaces hosting
- **Embedding Model:** paraphrase-multilingual-MiniLM-L12-v2 (384-dim, 50+ languages, local)
- **Search:** Hybrid BM25 + FAISS + RRF fusion + cross-encoder reranking
- **Agents:** LangGraph state machine (Plan → Execute → Reflect → Re-plan)
- **LLM:** Multi-provider support — OpenAI (GPT-4o-mini), Google Gemini, local Ollama. Configurable via LLM_PROVIDER env var.
- **Scoring weights:** Loaded from configs/scoring_weights.yaml at runtime, never hardcoded
- **TranslationPipeline is stubbed:** opus-mt models are ~300MB each per language pair. Loading 9 would exceed the hackathon's 16GB RAM / 5min CPU budget. The Redrob dataset (100K profiles) is all English anyway, so translation is a no-op for this dataset. Architecture is correct (lazy load, model lookup, fallback chain) but real model loading is disabled until needed for a non-English dataset.

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
- Phase 2 (synthetic data generation) is obsoleted by real dataset — plan should skip to Phase 3

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
- Session 4: Phase 1 implemented — settings.yaml, scoring_weights.yaml, models.yaml, config.py, constants.py, models.py
- Session 5: Phase 3 implemented — parser.py, normalizer.py, quality_scorer.py, extractor.py, tested on real Redrob dataset (100K profiles). Discovered real dataset in hackathon bundle, Phase 2 obsoleted.
- Session 6: Phase 4 implemented — language detector, translator pipeline, multilingual embedder
