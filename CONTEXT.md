# Project Context — India Runs

> Last updated: June 14, 2026 by Buffy (parent agent)
> ⚠️ Lines: 48/300 — keep under limit!

## Current Status
PRD complete, implementation plan complete, agent workflow rules established, git initialized with 2 commits. Multi-provider LLM support added (OpenAI, Gemini, Ollama). Ready to begin Phase 0 (environment setup). No code written yet.

## Active Tasks
- [ ] Phase 0: Environment setup (pyproject.toml, Docker, configs)
- [ ] Phase 1: Core infrastructure (config.py, models.py, constants.py, llm.py)
- [ ] Phase 2: Synthetic data generation (1,000 profiles, 50 queries)

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
| `configs/scoring_weights.yaml` | Tunable scoring weights (source of truth) |
| `configs/settings.yaml` | Application settings |

## Known Issues
- No code written yet — all planning docs only

## Environment
- Project path: `C:\Users\nanda\Desktop\india-runs`
- OS: Windows (win32), bash shell
- Python 3.11+ required
- Docker available for PostgreSQL + Redis
- No virtual environment set up yet

## Recent History
- Session 1: Researched hackathon, created PRD, IMPLEMENTATION_PLAN, agent rules, context/task log
- Session 2: Added multi-provider LLM support (OpenAI, Gemini, Ollama) to all docs
- Git: 2 commits — docs (PRD+plan) and chore (agent rules+context+gitignore)
