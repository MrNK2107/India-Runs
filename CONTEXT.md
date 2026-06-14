# Project Context — India Runs

> Last updated: June 14, 2026 by Buffy (parent agent)
> ⚠️ Lines: 48/300 — keep under limit!

## Current Status
PRD complete, implementation plan complete, agent workflow rules established. Ready to begin Phase 0 (environment setup) and Phase 1 (core infrastructure). No code written yet — all planning documents created.

## Active Tasks
- [ ] Phase 0: Environment setup (pyproject.toml, Docker, configs)
- [ ] Phase 1: Core infrastructure (config.py, models.py, constants.py)
- [ ] Phase 2: Synthetic data generation (1,000 profiles, 50 queries)
- [ ] Git init + first meaningful commit

## Architecture Decisions
- **UI Framework:** Gradio only (not Streamlit) — simpler for demos, free HuggingFace Spaces hosting
- **Embedding Model:** paraphrase-multilingual-MiniLM-L12-v2 (384-dim, 50+ languages, local)
- **Search:** Hybrid BM25 + FAISS + RRF fusion + cross-encoder reranking
- **Agents:** LangGraph state machine (Plan → Execute → Reflect → Re-plan)
- **LLM:** GPT-4o-mini for planner + rationale (cheapest, sufficient)
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
- Git repo not initialized yet
- No .gitignore created yet

## Environment
- Project path: `C:\Users\nanda\Desktop\india-runs`
- OS: Windows (win32), bash shell
- Python 3.11+ required
- Docker available for PostgreSQL + Redis
- No virtual environment set up yet

## Recent History
- Researched India Runs hackathon (Redrob AI, Track 1, ₹10L prize)
- Created PRD.md (24-section comprehensive requirements doc)
- Created IMPLEMENTATION_PLAN.md (78-file execution blueprint)
- Created .agent-rules.md (mandatory agent workflow rules)
- Created CONTEXT.md and TASK_LOG.md
