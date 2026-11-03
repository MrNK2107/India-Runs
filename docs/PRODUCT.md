# Kasauti — product brief

> **Kasauti** (कसौटी) is the touchstone jewellers in India rub gold against to show
> whether it is real. The product does the same for candidate profiles: it ranks them,
> then shows the streak, quoting the evidence behind every score.

**One line:** evidence-backed candidate ranking. Shortlists you can defend.

## Who it is for

| Persona | Situation | What they need |
|---|---|---|
| **In-house recruiter at an Indian scale-up** (50–2,000 people) | 300–3,000 applicants per role from Naukri, LinkedIn, referrals and career pages. Reviews by hand, or with an ATS keyword filter that misses half the good people. | A ranked list they trust enough to act on, and a way to say *why* candidate #4 beat #9 when the hiring manager asks. |
| **Hiring manager** | Gets a shortlist with no context and re-screens everyone. | To see, for each requirement, whether the candidate *proved* it or only *claimed* it. |
| **Staffing / RPO teams** | Submit candidates to clients and have to justify every submission. | An exportable shortlist with the reasoning attached. |
| **Talent platforms** (job boards, assessment platforms) | Have rich engagement and assessment signals but rank on keywords. | An API that blends those signals into ranking and returns explanations to show their own users. |

## The problem

1. **Keyword filters are wrong in both directions.** "Spark" in a summary passes the filter,
   while an engineer whose last two jobs were Spark pipelines, but who never filled in the
   skills box, gets filtered out. In the bundled sample, none of the top 5 results for
   *"data engineer with spark and airflow"* lists either skill in the skills section, yet every
   one of them shows both in their work history.
2. **AI rankers are black boxes.** A recruiter cannot forward "the model said 0.83" to a
   hiring manager, and a black box cannot be audited for bias or fraud.
3. **Profiles lie a little.** Inflated skill lists, impossible timelines (a job that starts
   before the company existed), "expert" in ten tools with zero years used.

## What Kasauti does

1. **Ranks** a talent pool against a role written in plain language: hybrid retrieval
   (FAISS vectors + BM25 keywords, fused with RRF), an optional cross-encoder reranker, then a
   10-signal weighted score (skills, semantic fit, engagement, experience, career trajectory,
   skill depth, and more).
2. **Explains** every result:
   - **Score receipt.** Each signal's weight × score as points of the final score; the
     points add up to the number shown, and the role-fit multiplier is spelled out.
   - **Claims vs proof.** Every skill, location, experience band and role in the query is
     checked against the profile and graded: *verified* (platform assessment), *listed*
     (skills section), *used in a role* (title or description of an actual job), *only claimed*
     (summary text), or *missing*. Each grade quotes the sentence it came from, with the match
     highlighted.
   - **Strengths and risks.** Responsiveness, notice period, passive status, short stints,
     integrity flags, and skills that are claimed but unproven.
3. **Checks itself.** Each result list gets a fairness check: score gaps by city tier,
   institute and profile language, flagged above 10 points. Any candidate can be explained
   against the query by ID ("why isn't she on the list?").
4. **Compares.** A shortlist turns into a requirement matrix (who proves what), and exports
   to CSV with the reasoning attached.
5. **Stays live.** `POST /ingest` (or drag-and-drop in the UI) validates, persists, embeds and
   indexes new profiles; the next search can rank them. Re-uploads are idempotent.

## Differentiators

| | Kasauti | ATS keyword filter | Black-box AI matcher |
|---|---|---|---|
| Finds skills used in past jobs but missing from the skills list | ✅ shown as evidence; can be credited in the score (opt-in) | ❌ | sometimes, silently |
| Tells a claim apart from proof | ✅ 4-level evidence grade with quotes | ❌ | ❌ |
| Score you can audit | ✅ points per signal, sums to the score | n/a | ❌ |
| Fraud / anomaly flags | ✅ impossible timelines, skill-density and expert-with-zero-years checks; ×0.15 penalty, shown as a risk | ❌ | rarely |
| Platform engagement signals in ranking | ✅ response rate, interview completion, verification, GitHub activity, … | ❌ | varies |
| Runs with no model or GPU | ✅ lite engine, under 300 MB RAM | ✅ | ❌ |
| India-specific | Hinglish queries, Indian company and city knowledge, Bengaluru/Bangalore folding | ❌ | ❌ |

## Ranking choices left to the user

Two places where a more generous reading of the profile changes the ranking. Each ships as
an option, with today's ranking as the default:

- **`skill_evidence`**: `listed` (default) scores skills from the skills section only;
  `work_history` also counts skills proven in a role or an assessment. Summary-only claims
  never count.
- **`trajectory_mode`**: `completed_roles` (default) measures job stability on finished roles;
  `include_current` also counts the tenure of the open role, so a long current job is no longer
  ignored.

Both are available in the API and under "Tune the ranking" in the UI.

## Engines

| | Lite (default in Docker) | Full |
|---|---|---|
| Embeddings | deterministic feature hashing (word, bigram, char-trigram) | multilingual MiniLM (50+ languages) |
| Reranker | hybrid scores | ms-marco cross-encoder |
| Agents | deterministic turbo pipeline | LangGraph plan → execute → listwise rank → reflect (needs an LLM key) |
| RAM | ~0.3 GB | 2–4 GB |
| Use for | demos, free-tier hosting, CI, reproducible rankings | production quality on large pools |

## Positioning and name

- **Category:** explainable talent ranking (an API, plus a recruiter workspace on top).
- **Why rename "India Runs":** that was the hackathon track name. It says nothing about what the
  product does, and it sounds like a fitness event. *Kasauti* is short and Indian, and it carries
  the core promise: test the claim, show the proof. The repository folder and package name stay
  `india-runs`.
- **Voice:** plain and specific. We say "used Spark in a role at Tech Mahindra", not
  "AI-powered talent intelligence".

## Roadmap

1. Resume PDF intake via the existing (tested, not yet wired) LLM `FieldExtractor`.
2. Per-role saved searches with shortlist history and notes.
3. Fairness *mitigation* options (blind mode that hides name, institute and city in the UI)
   to go with the per-search fairness check.
4. ATS connectors (Greenhouse, Lever, Zoho Recruit) to push shortlists back.
5. Recruiter feedback (thumbs up/down on evidence) as training signal for the reranker.
