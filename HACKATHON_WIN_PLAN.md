# Hackathon Win Plan — India Runs

## Strategy: 4 Tracks Aligned to Judging Criteria

| Track | Focus | Judging Weight | Effort |
|-------|-------|---------------|--------|
| **A** | Make it actually run end-to-end | Technical Execution (25%) | Critical |
| **B** | Make it look like a real product | Presentation (25%) | High |
| **C** | Add the differentiators that win | Innovation (25%) | Medium |
| **D** | Prove real-world impact | Real-world Impact (25%) | Medium |

---

## Track A: Make It Actually Run (Critical Path)

### A1 — Fix `build_indexes.py` to work with the real data
**Files:** `scripts/build_indexes.py`, `src/core/constants.py`

- `PROFILES_PATH` points to `profiles.json` but real file is `candidates.jsonl` → fix path
- Script uses `json.load()` → will OOM on 487MB. Use the streaming `parse_jsonl_file()` from parser
- Add `--sample N` flag for quick dev (only process first N profiles)
- Add `--force` flag to rebuild even if indexes exist

### A2 — Add synthetic ground truth + queries for evaluation
**Files:** `scripts/generate_data.py` (update), `scripts/evaluate.py`

- `evaluate.py` needs `queries.json` + `ground_truth.json` — neither exists
- Generate 10 sample queries with profile_id ground truth mapping from the sample data
- Make evaluation run with or without real indexes (graceful skip)

### A3 — Wire up the Gradio UI to actually return results
**Files:** `src/ui/app.py`, `src/api/routes/search.py`

- The search handler imports `_orchestrator` from `routes/search.py` — verify it's initialized on startup
- Add `init_orchestrator()` wiring in `src/main.py` lifespan
- Add loading state, error handling, and empty state to the UI

### A4 — Docker startup that works
**Files:** `Dockerfile`, `docker-compose.yml`

- Build indexes on container start if missing
- Expose Gradio UI port
- One-command startup

---

## Track B: Make It Look Like a Real Product (Presentation)

### B1 — Scoring Slider UI (FR-7.2a)
**Files:** `src/ui/app.py`, `src/ui/components.py`, `src/ui/styles.css`

The PRD calls for interactive sliders with 6 recruiter-facing dimensions:
- Skill match, Experience, Education, Assessment, Behavioral, Cultural fit
- Sliders update score in real-time
- Color-coded final score (red < 40, yellow 40-60, blue 60-80, green > 80)
- Formula display: `score = (s1×0.30) + (s2×0.25) + ...`

**Why this wins:** Judges love interactive demos. This shows the scoring is *configurable*, not a black box.

### B2 — Live Analytics Dashboard
**Files:** `src/ui/components.py`

- Fairness metrics with visual badges (DIR ratio, bias flags)
- Score distribution histogram
- Language distribution pie chart
- Latency p50/p95/p99

### B3 — Better Candidate Cards + Rationale
**Files:** `src/ui/components.py`

- Score badge with color (red/yellow/blue/green)
- Skills as chips (green = matched, red = missing)
- Radar chart showing all 6 scoring dimensions
- Pros/cons list per candidate

### B4 — Dark Mode
**Files:** `src/ui/styles.css`

- auto-detects via `prefers-color-scheme`
- Toggle button in UI

---

## Track C: Innovation Differentiators

### C1 — Plackett-Luce Listwise Ranking
**Files:** `src/ranking/listwise_ranker.py` (new), `src/agents/orchestrator.py`

Currently: pointwise ranking (each candidate scored independently).
Implement: candidates compete in tournament-style groups, producing globally coherent rankings.

**Why this wins:** Most HR tech uses pointwise scoring. Listwise ranking is an academic-grade innovation. Namedropping "Plackett-Luce" in the pitch is a judge magnet.

### C2 — PII Anonymization
**Files:** `src/fairness/anonymizer.py` (new), `src/fairness/bias_detector.py`

Strip names, universities, locations before any LLM evaluation. Show side-by-side comparison: "With anonymization vs without."

**Why this wins:** Amazon's AI hiring tool was famously biased. Showing you *built a fix* for the industry's biggest scandal is powerful storytelling.

### C3 — Code-mixed Hindi queries work in demo
**Files:** `src/ui/app.py` (add Hindi example queries)

Add these as example chips:
- "Mujhe ek DevOps engineer chiye jisko AWS aur Kubernetes mein 5 saal ka experience hai"
- "Python aur data science mein 3 saal ka anubhav wala candidate dhoondhein"

**Why this wins:** Real Indian market relevance. Most teams won't handle Hinglish.

---

## Track D: Real-World Impact

### D1 — Fairness Badge on Every Search
**Files:** `src/ui/app.py`, `src/fairness/metrics.py`

After every search, compute fairness metrics and show a badge:
- ✅ Fair (DIR ≥ 0.80)
- ⚠️ Review Recommended (DIR 0.70-0.80)
- 🚫 Bias Detected (DIR < 0.70)

### D2 — Generate the Submission CSV
**Files:** `scripts/generate_submission.py` (new)

Generate a `team_xxx.csv` with 100 ranked candidates + reasoning per the hackathon spec.

### D3 — One-command demo
```bash
docker compose up
```
Opens Gradio at localhost:7860 with pre-built indexes, sample queries, and live search.

---

## Execution Order

```
Phase 1 (Critical): A1 + A2 → Get indexes building + evaluation running
Phase 2 (Critical): A3 → Get the UI returning real results
Phase 3 (High): B1 + B3 → Scoring sliders + better candidate cards
Phase 4 (High): C1 + C2 → Plackett-Luce + PII anonymization
Phase 5 (Medium): B2 + D1 → Live analytics + fairness badges
Phase 6 (Medium): B4 + C3 → Dark mode + Hindi examples
Phase 7 (Polish): A4 + D2 + D3 → Docker + submission CSV + one-command demo
```

---

## What the Judge Sees (The Pitch)

When they open the demo:
1. **Search tab**: Type "senior DevOps engineer 5 years AWS" → results appear in < 2s
2. **Candidate cards**: Show score badge, skill chips (green/red), radar chart
3. **Scoring sliders**: Drag "Skill Match" slider → results re-rank in real-time
4. **Rationale panel**: Click a candidate → see "Why this candidate #1" with strengths, gaps, evidence
5. **Analytics tab**: Fairness dashboard showing DIR = 0.92 ✅, no name/language/university bias
6. **Hindi query**: Click "Mujhe ek DevOps engineer chiye..." → results in Hindi
7. **Dark mode toggle**: Because it looks cool

**The story:** "We built an AI recruiter that doesn't just match keywords — it understands context, catches its own mistakes, checks for bias, and explains every decision. And it works in Hinglish."
