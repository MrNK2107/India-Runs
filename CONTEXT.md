# Project Context — India Runs

> Last updated: June 17, 2026 by opencode agent
> Lines: 52/300

## Current Status
**Semantic skill matching** — skill_match now falls back to `raw_text` when structured `profile.skills` is incomplete. Normalizer also extracts skills from text during ingestion. This solves the core problem: candidates found via semantic search (80%) no longer get penalized to 0% skill_match just because structured data is missing.

## Active Tasks
- [x] Skill extraction from raw_text during normalization (normalizer.py)
- [x] raw_text fallback in _match_skills_detail / _skill_match_score (executor.py)
- [ ] Phase 3: Fairness dashboard + submission polish

## Architecture Decisions
- **Skill matching = structured + unstructured:** `_match_skills_detail` first checks `profile.skills` (exact → alias → fuzzy), then falls back to `profile.raw_text` (substring + alias). This means skill_match reflects the full profile, not just one field.
- **Normalizer enriches skills at ingestion:** Scans raw_text for known skill names from SKILL_ALIASES and adds any missing ones to `profile.skills` with confidence=0.6. This seeds structured data for future reuse.
- **Simple query threshold = 3:** Keeps most queries routed through LLM planner for deeper understanding.
- **Cross-encoder normalization:** Sigmoid maps raw logits → [0,1].
- **LLM timeout:** 15s asyncio timeout on planner calls to prevent hanging.

## Key Files
| File | Purpose |
|------|---------|
| `src/ingestion/normalizer.py` | Extracts skills from raw_text, adds to profile.skills |
| `src/agents/executor.py` | Skill matching with raw_text fallback, aliases, fuzzy |
| `src/agents/orchestrator.py` | Query routing, parsing, stop-word filter, location extraction |

## Known Issues
- Queries >3 words through LLM planner — Ollama must be running or 15s timeout fallback
- Auto-ground-truth is self-referential

## Environment
- LLM_PROVIDER=ollama, OLLAMA_MODEL=qwen2.5:7b

## Recent History
- Session 23: Semantic skill matching — raw_text fallback + normalizer extraction. Solves "0% skill_match despite 80% semantic" contradiction.
