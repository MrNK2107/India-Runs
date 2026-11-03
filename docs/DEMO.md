# Kasauti — 5-minute demo script

**Setup (before the audience arrives):**

```bash
docker compose up --build          # or: EMBEDDING_BACKEND=hashing CROSS_ENCODER_ENABLED=false uvicorn src.main:app --port 7860
```

Open http://localhost:7860. The status pill in the top right should read
`500 profiles · lite engine · sample · sample_500.jsonl` with a green dot. To start each run
with a fresh pool, delete `data/profiles/ingested.jsonl` (or the `profiles` compose volume).

---

### 1. The problem (30 s)

> "A recruiter has 500 applicants for a data engineering role. A keyword filter keeps whoever
> typed *Spark* into the skills box. An AI matcher gives you 0.83 and no reason. Neither one
> holds up when the hiring manager asks *why this person?*"

### 2. Rank with receipts (60 s)

Click the example chip **"data engineer with spark and airflow in bangalore"**.

- Point at the **rank bar** on card #1: "This bar *is* the score. Each colour is one signal's
  share of the points."
- On the right, **Where the score comes from**: the points add up to the score.
  "Notice *Required skills: +0.0*. Hold that thought."

### 3. Claims vs proof (60 s)

Scroll the right panel to **Requirements, with evidence**.

- *spark → MET, used in role*: "Kasauti quotes the sentence. Siya used Spark at Tech Mahindra;
  here it is, highlighted."
- The dashed **only claimed** quote below it: "The summary also says Spark. A claim is not proof,
  and Kasauti grades the two differently."
- **Strengths**: "Proven in work history but missing from the skills list. That's why the
  skills score was zero: the ranker only reads the skills box."

### 4. Fix it, deliberately (45 s)

Open **Tune the ranking** → tick **Credit skills proven in work history** → **Rank candidates**.

- Siya goes from 49 (*potential*) to 69 (*good*), and her rank bar gains a gold skills
  segment. Without the city filter, the pool's top four data engineers all move from about 50
  to about 70 (*strong*).
- "It's an explicit option with the default unchanged. The same goes for counting current-role
  tenure in career stability. Ranking policy belongs to the hiring team, not to the model."

### 5. Risks and the shortlist (45 s)

- Click through a few candidates and read **Risks**: *90-day notice period*, *Rarely replies to
  recruiters*, *Claimed but unproven: airflow*. Struck-through red chips on a card mark
  requirements with no evidence at all.
- Star 3–4 candidates → **Shortlist**: "The requirement matrix shows who proves what at a
  glance." → **Export CSV**: "It goes to the hiring manager with the reasoning attached."

### 5b. Why not them? (20 s)

Below the list, type `CAND_0000001` into **Not seeing someone?** → **Explain**.

- "Ira isn't on the list. Here's why: a different track (Backend Engineer, role-fit ×0.79),
  even though she uses Spark in two roles." Point at the **career timeline**, where the roles
  marked *proof* hold the evidence.
- Above the list, the **Fairness check** reads *no gap* for city tier and institute: "Every
  list is checked for score gaps between tier-1 and tier-2/3 cities and between IIT/NIT/BITS and
  other institutes."

### 6. Live talent pool (45 s)

Search **"flutter developer with dart and firebase"**: Java and .NET developers fill the list.
Click **Add profiles** → **Use the example profile**.

- "Validated, stored, embedded and indexed." The list re-ranks on its own: **Asha Rao, Flutter
  Developer, Pune** is #1 with a `new` badge and *strong* verdict.
- Upload the same file again: `1 unchanged`. Ingest is idempotent.

### 7. Close (15 s)

> "No GPU, no API key, 300 MB of RAM. The same API runs the full transformer and LLM-agent
> stack when you need it. Every ranking comes with receipts."

---

**API version of the demo** (for technical audiences):

```bash
curl -s localhost:7860/api/v1/health | jq .engine
curl -s localhost:7860/api/v1/search -H 'content-type: application/json' \
  -d '{"query":"data engineer with spark and airflow","max_results":3}' \
  | jq '.results[] | {rank, name, score: .scores.overall, why: .explanation.headline}'
curl -s localhost:7860/api/v1/explain -H 'content-type: application/json' \
  -d '{"query":"data engineer with spark and airflow","profile_id":"CAND_0000001"}' | jq '.risks'
curl -s -X POST localhost:7860/api/v1/ingest -F "file=@data/samples/ingest_example.json;type=application/json"
```

**If something goes wrong:** a grey dot on the status pill means the API is not up (check the
server log). An empty result list means the filters are too strict: clear the city and min-years
fields under *Tune the ranking*.
