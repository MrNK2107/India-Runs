# Project Context — India Runs

> Last updated: June 17, 2026 by opencode agent
> Lines: 88/300

## Current Status
Pipeline runs end-to-end with Ollama (qwen2.5:7b) for intelligent planning/reflection. ML models pre-loaded at server startup (no 40s first-request delay). Ground truth generated from 50 sample profiles (50 queries, 108 relevance labels). Evaluation reports real precision/recall/NDCG/MRR metrics. No demo mode. 86 tests pass, ruff clean, 0 warnings.

## Active Tasks
- [x] Ollama integration (qwen2.5:7b) — LLM_PROVIDER set, local planner + reflector
- [x] Pre-load embedding + cross-encoder models in lifespan
- [x] Generate ground truth from sample profiles (50 queries)
- [x] Rewrite evaluate.py — real metrics, no demo mode
- [ ] Phase 2: Index full 100K dataset
- [ ] Phase 3: Fairness dashboard + submission polish

## Architecture Decisions
- **LLM:** Ollama (qwen2.5:7b) — local, zero API costs, 32K context
- **Model pre-loading:** Both embedder and cross-encoder models are loaded at server startup (accessing `.model` property in lifespan), eliminating the 40s first-request delay
- **Ground truth:** Auto-generated from profiles. For each profile → extract headline + skills → create query → mark profile as relevant. Clusters by skill Jaccard similarity (threshold 0.2) to add related profiles as relevant.
- **Evaluation:** 50 queries with ground truth. Metrics: precision@k, recall@k, MRR, NDCG, cross-lingual MRR. Report saved to `data/evaluation_report.json`.

## Key Files
| File | Purpose |
|------|---------|
| `scripts/generate_ground_truth.py` | Generates queries + relevance labels from profiles |
| `scripts/evaluate.py` | Full evaluation with real ground truth (no demo mode) |
| `src/main.py` | Pre-loads ML models at startup |
| `data/queries/queries.json` | 50 generated queries |
| `data/ground_truth/ground_truth.json` | 50 entries with relevance labels (108 total) |
| `data/evaluation_report.json` | Latest evaluation results |

## Known Issues
- Pip dependency conflict with supabase packages (httpx<0.28) — unrelated
- First model download still takes ~8s on cold start (with HF_TOKEN). Subsequent runs use cache.
- Auto-ground-truth is self-referential (profile → query → same profile as relevant). Since there's no real labeled dataset, this is the best available approach for objective metrics.

## Environment
- LLM_PROVIDER=ollama, OLLAMA_MODEL=qwen2.5:7b
- HF_TOKEN set in .env — faster HF model downloads

## Recent History
- Session 20: Ollama integration, model pre-loading, ground truth generation, real evaluation. Phase 1 complete.
