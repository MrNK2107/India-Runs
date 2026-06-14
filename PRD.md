# Product Requirements Document (PRD)
# Intelligent Candidate Discovery System — India Runs Track 1

> **Hackathon:** India Runs by Redrob AI — Track 1: Data & AI Challenge  
> **Prize Pool:** ₹10 Lakhs (₹2,00,000 Grand Champion)  
> **Platform:** Hack2Skill  
> **Event Duration:** 42 days (Registration closes June 28, 2026; Event ends July 31, 2026)  
> **PRD Version:** 2.0  
> **Last Updated:** June 14, 2026 (updated: comprehensive gap analysis + winning strategies from Architectural Blueprint)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Success Metrics](#3-goals--success-metrics)
4. [User Personas](#4-user-personas)
5. [System Architecture](#5-system-architecture)
6. [Data Requirements](#6-data-requirements)
7. [Data Models & Schemas](#7-data-models--schemas)
8. [Core Features — Functional Requirements](#8-core-features--functional-requirements)
9. [API Specifications](#9-api-specifications)
10. [Agentic Workflow — Detailed Design](#10-agentic-workflow--detailed-design)
11. [Multilingual Processing Pipeline](#11-multilingual-processing-pipeline)
12. [Ranking & Matching Engine](#12-ranking--matching-engine)
13. [Rationale Generation System](#13-rationale-generation-system)
14. [Bias Mitigation & Fairness](#14-bias-mitigation--fairness)
15. [UI/UX Requirements](#15-uiux-requirements)
16. [Non-Functional Requirements](#16-non-functional-requirements)
17. [Tech Stack — Pinned Versions](#17-tech-stack--pinned-versions)
18. [Project Structure](#18-project-structure)
19. [Implementation Phases & Timeline](#19-implementation-phases--timeline)
20. [Testing Strategy](#20-testing-strategy)
21. [Deployment Requirements](#21-deployment-requirements)
22. [Submission Package](#22-submission-package)
23. [Risk Assessment](#23-risk-assessment)
24. [Appendix](#24-appendix)

---

## 1. Executive Summary

Build an **Intelligent Candidate Discovery System** that goes beyond traditional keyword-based Applicant Tracking Systems (ATS). The system uses **hybrid semantic search** (vector + keyword), **agentic AI workflows** (plan → execute → reflect), and **multilingual NLP** to match candidates to job roles with high precision, explainability, and fairness — specifically tuned for the Indian hiring landscape where profiles are messy, multilingual, and spread across 50+ platforms.

**What makes this a winner:**
- Hybrid retrieval (BM25 + dense embeddings + cross-encoder reranking) outperforms pure vector search by ~25%
- Agentic architecture with reflection/critique loop catches false positives before they reach the recruiter
- Multilingual support (30+ Indian languages) — aligned with Redrob AI's core mission
- Rationale reports for every match — not just a ranked list
- Bias-aware ranking that matches on skills, not proxies
- **Listwise tournament ranking** (Plackett-Luce) — candidates compete head-to-head in tournament rounds for globally coherent rankings
- **PII redaction layer** — names, universities, locations stripped before LLM evaluation to prevent name-based and institution-based bias
- **Scoped pre-search retrieval** — structural filters narrow the candidate pool before expensive vector search runs, solving Vector Search Dilution at scale
- **Multi-dimensional YAML rationale** — 12-20 granular dimensions per candidate (not just 6)

---

## 2. Problem Statement

### The Core Problem
Traditional ATS systems rely on **keyword matching**. A candidate who writes "managed cloud infrastructure" gets rejected for a "DevOps Engineer" role because the keyword "DevOps" doesn't appear — even though their experience is directly relevant.

### Specific Challenges in the Indian Market
1. **Multilingual Profiles:** Candidates write resumes in Hindi, Tamil, Telugu, Marathi, Bengali, etc. — not just English. Keyword search fails across languages.
2. **Unstandardized Data:** Profiles are scraped from 50+ platforms (LinkedIn, Naukri, AngelList, GitHub, company career pages) with inconsistent formats, missing fields, and noisy data.
3. **Passive Talent:** ~70% of the workforce is passive (not actively applying). Traditional search misses them because it only indexes active applicants.
4. **False Positives:** A pure semantic search might match a "Marketing Manager" to a "Product Manager" role because the embeddings are too similar. Precision matters.
5. **Bias:** AI systems can perpetuate hiring bias by overweighting proxies (university pedigree, name-based ethnicity inference, gendered language).

### What Redrob AI Specifically Needs
Based on Redrob's platform capabilities:
- A system that can search across their **700M+ profile database**
- Handle **30+ languages** natively (not just bolted-on translation)
- Understand **Indian hiring context** (salary benchmarks in tier-2 cities, local career progression norms)
- Provide **contextual signals** (job changes, skill updates, open-source contributions)
- Operate at **<50ms response times** for production feasibility

---

## 3. Goals & Success Metrics

### Primary Goals
| Goal | Metric | Target |
|------|--------|--------|
| Semantic Matching Quality | Precision@10 | ≥ 0.85 |
| Semantic Matching Quality | Recall@50 | ≥ 0.90 |
| Cross-lingual Matching | Cross-lingual MRR | ≥ 0.75 |
| Latency | End-to-end response time | < 2 seconds (demo), < 500ms (search only) |
| Rationale Quality | Human evaluation (1-5 scale) | ≥ 4.0 |
| Multilingual Coverage | Languages supported | ≥ 10 Indian languages |
| False Positive Rate | Incorrect matches in top-10 | ≤ 10% |

### Secondary Goals
| Goal | Metric | Target |
|------|--------|--------|
| Passive Candidate Discovery | % of passive candidates surfaced | ≥ 40% |
| Bias Audit | Disparate impact ratio | ≥ 0.80 (4/5ths rule) |
| User Satisfaction | Demo usability score | ≥ 4.2/5.0 |
| Code Quality | Test coverage | ≥ 80% |
| Documentation | API docs + README completeness | 100% |

### Judging Alignment
Hackathon judges will evaluate on:
1. **Innovation** (25%) — Agentic architecture, hybrid search, rationale generation
2. **Technical Execution** (25%) — Clean code, proper ML pipeline, working demo
3. **Real-world Impact** (25%) — Handles Indian market realities, multilingual, passive talent
4. **Presentation** (25%) — Pitch deck, live demo, clear metrics

---

## 4. User Personas

### Persona 1: Technical Recruiter (Primary)
- **Name:** Priya
- **Role:** Senior Technical Recruiter at a Series B startup
- **Needs:** Find 5 DevOps engineers in Bangalore with 3+ years experience who have scaled infrastructure
- **Pain Point:** Current ATS filters out great candidates who don't use exact keywords
- **Wants:** A ranked list with *explanation* of why each candidate matches
- **Languages:** English, Hindi, Kannada

### Persona 2: Hiring Manager (Secondary)
- **Name:** Arjun
- **Role:** VP Engineering
- **Needs:** Review shortlist quality, understand why candidates were selected
- **Pain Point:** Doesn't trust AI recommendations without reasoning
- **Wants:** Confidence scores with breakdown (skill match %, experience match %, culture fit indicators)
- **Languages:** English, Hindi

### Persona 3: HR Generalist (Tertiary)
- **Name:** Meera
- **Role:** HR Manager at a mid-size company
- **Needs:** Fill 3 different roles across departments simultaneously
- **Pain Point:** Limited technical knowledge, can't evaluate candidates herself
- **Wants:** Simple natural language search ("Find someone who can manage our AWS costs")

---

## 5. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│              (Gradio / Streamlit Demo App)                        │
│                                                                   │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐       │
│  │ Job Query │  │ Results View │  │ Rationale Inspector   │       │
│  │ Input     │  │ + Rankings   │  │ (per-candidate why)   │       │
│  └─────┬────┘  └──────▲───────┘  └──────────▲───────────┘       │
└────────┼───────────────┼────────────────────┼────────────────────┘
         │               │                    │
         ▼               │                    │
┌────────────────────────┼────────────────────┼────────────────────┐
│                   API LAYER (FastAPI)                            │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐        │
│  │ /search      │  │ /candidates │  │ /rationale        │        │
│  │ POST         │  │ GET by ID   │  │ GET for match     │        │
│  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘        │
└─────────┼────────────────┼──────────────────┼────────────────────┘
          │                │                   │
          ▼                │                   │
┌─────────────────────────────────────────────────────────────────┐
│                   AGENTIC ORCHESTRATOR                           │
│                    (LangGraph State Machine)                     │
│                                                                   │
│  ┌──────────┐    ┌───────────┐    ┌───────────┐                │
│  │ PLANNER  │───▶│ EXECUTOR  │───▶│ REFLECTOR │──┐             │
│  │ Agent    │    │ Agent     │    │ (Critic)  │  │             │
│  └──────────┘    └───────────┘    └─────┬─────┘  │             │
│       ▲                                 │        │              │
│       │         ◀───────────────────────┘        │              │
│       │    (Re-plan if critique fails)           │              │
│       └──────────────────────────────────────────┘              │
│                                                                   │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                  TOOLS                               │        │
│  │  ┌────────────┐ ┌──────────┐ ┌────────────────┐   │        │
│  │  │ Hybrid     │ │ Multilingual│ │ Profile       │   │        │
│  │  │ Search     │ │ Processor │ │ Enricher      │   │        │
│  │  │ (BM25+Vec) │ │           │ │               │   │        │
│  │  └────────────┘ └──────────┘ └────────────────┘   │        │
│  └─────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
          │                │                   │
          ▼                ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ FAISS    │  │ BM25     │  │ PostgreSQL│  │ Redis    │       │
│  │ Vector   │  │ Index    │  │ Profile   │  │ Cache    │       │
│  │ Store    │  │ (rank_bm25)│ │ Store    │  │          │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Embedding Model Cache                     │       │
│  │  paraphrase-multilingual-MiniLM-L12-v2                 │       │
│  │  (local, no API calls needed)                          │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Diagram (3-Stage Pipeline)

```
              USER QUERY
                  |
                  v
    +-------------------+      +-------------------+
    |   Stage 1a: FTS   |      |   Stage 1b: Vec   |
    |  BM25 Sparse Index|      |  FAISS Dense Retr |
    +-------------------+      +-------------------+
              |                         |
              +---------> RRF <---------+
                  (parallel execution)
                          |
                          v
                +-------------------+
                |  STRUCTURAL PRE-  |  <-- Scoped retrieval: filters BEFORE vector
                |  SEARCH FILTERS   |      search narrows the candidate pool
                +-------------------+
                          |
                          v
                +-------------------+
                |   Stage 2: Cross  |
                |  Encoder Reranker |
                +-------------------+
                          |
                          v
               +---------------------+
               |   Stage 3:          |
               |  LangGraph Agent    |
               |  Plan->Exec->Reflect|
               +---------------------+
                          |
                          v
               +---------------------+
               |  Listwise           |
               |  Tournament Ranking |
               |  (Plackett-Luce)    |
               +---------------------+
                          |
                          v
             RATIONALE (12-20 dims) + PII REDACTED OUTPUT
```

### Architecture Principles
1. **Offline-heavy:** All heavy computation (embedding, indexing) happens at build time. Runtime is fast retrieval + LLM generation.
2. **Modular:** Each component (parser, embedder, searcher, ranker, generator) is independently testable and replaceable.
3. **Fail-safe:** If any component fails, fall back gracefully (e.g., if cross-encoder is slow, skip reranking; if LLM is unavailable, return raw matches without rationale).
4. **Local-first:** Use local models where possible to minimize API costs and latency. LLM calls only for planning and rationale generation.
5. **Provider-agnostic:** LLM layer supports OpenAI, Google Gemini, and local Ollama. Configuration determines which provider is used. All agents use a unified interface.
6. **Scoped pre-search filtering:** Structural filters (location, experience, company) narrow the candidate pool before FAISS/BM25 search, not after — solving Vector Search Dilution at scale.
7. **Parallel execution:** BM25 and FAISS searches run concurrently (not sequentially) to minimize latency.
8. **PII-free evaluation pipeline:** Candidate names, institutions, and locations are stripped before any LLM evaluation to prevent name-based and institution-based bias.
9. **Listwise over pointwise:** Final ranking uses a Plackett-Luce tournament model where candidates compete in small groups, producing globally coherent rankings rather than independent pointwise scores.

---

## 6. Data Requirements

### 6.1 Input Data Sources

The system must handle profiles from multiple sources with varying formats:

| Source | Format | Fields Available | Quality |
|--------|--------|-----------------|---------|
| LinkedIn | JSON/CSV export | name, title, summary, skills, experience, education, location | High |
| Naukri.com | Scraped HTML/JSON | name, title, skills, experience, education, location, salary | Medium |
| GitHub | API JSON | username, repos, languages, contributions, README | High |
| Resume PDFs | Unstructured text | everything (parsed via NLP) | Low-Medium |
| Company career pages | HTML/JSON | varies wildly | Low |

### 6.2 Profile Schema (Normalized)

Every profile, regardless of source, must be normalized to this schema:

```json
{
  "profile_id": "string (UUID)",
  "source": "linkedin | naukri | github | resume_pdf | career_page | manual",
  "raw_text": "string (constructed from fields, see Section 6.2a)",

  "personal": {
    "name": "string",
    "location": {
      "city": "string | null",
      "state": "string | null",
      "country": "string",
      "is_remote_ok": "boolean"
    },
    "languages_spoken": ["string"],
    "native_language": "string | null"
  },

  "professional": {
    "current_title": "string | null",
    "current_company": "string | null",
    "total_experience_years": "float | null",
    "industry": "string | null",
    "employment_type": "full_time | part_time | contract | freelance | student | null"
  },

  "skills": [
    {
      "name": "string (e.g., 'Python', 'AWS EC2')",
      "category": "programming_language | framework | tool | soft_skill | domain_knowledge | certification",
      "proficiency": "beginner | intermediate | advanced | expert | null",
      "years_used": "float | null",
      "evidence": "string (snippet from profile proving this skill)",
      "confidence": "float (0.0-1.0, how confident we are this skill exists)"
    }
  ],

  "experience": [
    {
      "title": "string",
      "company": "string",
      "start_date": "YYYY-MM | null",
      "end_date": "YYYY-MM | null",
      "is_current": "boolean",
      "description": "string",
      "highlights": ["string"],
      "skills_demonstrated": ["string"],
      "location": "string | null"
    }
  ],

  "education": [
    {
      "institution": "string",
      "degree": "string | null",
      "field": "string | null",
      "start_date": "YYYY | null",
      "end_date": "YYYY | null",
      "gpa": "float | null"
    }
  ],

  "signals": {
    "is_passive": "boolean (not actively job-seeking)",
    "last_active_date": "YYYY-MM-DD | null",
    "open_to_work": "boolean | null",
    "github_activity_score": "float (0.0-1.0) | null",
    "has_portfolio": "boolean",
    "certifications": ["string"],
    "publications": ["string"],
    "speaking_engagements": ["string"]
  },

  "metadata": {
    "language_detected": "string (ISO 639-1)",
    "original_language": "string",
    "was_translated": "boolean",
    "translation_confidence": "float | null",
    "embedding_vector_id": "int (index in FAISS)",
    "bm25_doc_id": "int (index in BM25 index)",
    "created_at": "ISO 8601 datetime",
    "updated_at": "ISO 8601 datetime",
    "data_quality_score": "float (0.0-1.0)"
  }
}
```

### 6.2a `raw_text` Construction

The `raw_text` field is constructed by concatenating profile fields in this order:

```
"Name: {name}. Title: {current_title}. Company: {current_company}. "
"Summary: {summary}. Skills: {skill_names joined by ', '}. "
"Experience: {each experience as 'Title at Company (start-end): description; skills_demonstrated'}. "
"Education: {each education as 'Degree in Field at Institution'}. "
"Certifications: {certifications joined by ', '}. "
"Languages: {languages_spoken joined by ', '}.
```

This text is used for:
1. **Embedding** — fed into the multilingual embedding model
2. **BM25 indexing** — tokenized for keyword search
3. **LLM context** — summarized for rationale generation

### 6.2b `data_quality_score` Computation

```python
def compute_data_quality_score(profile: dict) -> float:
    """
    Returns 0.0-1.0 based on completeness and noise.
    
    Scoring breakdown:
    - Name present: +0.10
    - Title present: +0.10
    - At least 1 skill: +0.15
    - At least 1 experience: +0.15
    - Education present: +0.10
    - Location present: +0.10
    - raw_text length > 200 chars: +0.10
    - raw_text length > 500 chars: +0.05
    - No obvious encoding artifacts: +0.05
    - Skills have evidence snippets: +0.10
    
    Returns: sum of applicable bonuses (0.0 - 1.0)
    """
    score = 0.0
    if profile.get("personal", {}).get("name"): score += 0.10
    if profile.get("professional", {}).get("current_title"): score += 0.10
    if profile.get("skills"): score += 0.15
    if profile.get("experience"): score += 0.15
    if profile.get("education"): score += 0.10
    if profile.get("personal", {}).get("location", {}).get("city"): score += 0.10
    raw = profile.get("raw_text", "")
    if len(raw) > 200: score += 0.10
    if len(raw) > 500: score += 0.05
    if not has_encoding_artifacts(raw): score += 0.05
    if any(s.get("evidence") for s in profile.get("skills", [])): score += 0.10
    return min(score, 1.0)
```

### 6.3 Job Query Schema

```json
{
  "query_id": "string (UUID)",
  "raw_query": "string (natural language from recruiter)",
  "parsed": {
    "required_skills": [
      {
        "name": "string",
        "importance": "required | preferred | nice_to_have",
        "min_proficiency": "beginner | intermediate | advanced | expert | null",
        "min_years": "float | null"
      }
    ],
    "preferred_skills": [
      {
        "name": "string",
        "importance": "nice_to_have",
        "weight": "float (0.0-1.0)"
      }
    ],
    "experience": {
      "min_years": "float | null",
      "max_years": "float | null",
      "industry": "string | null"
    },
    "location": {
      "city": "string | null",
      "state": "string | null",
      "country": "string | null",
      "remote_ok": "boolean",
      "hybrid_ok": "boolean"
    },
    "education": {
      "min_degree": "string | null",
      "field": "string | null"
    },
    "salary": {
      "min": "float | null",
      "max": "float | null",
      "currency": "INR"
    },
    "filters": {
      "exclude_companies": ["string"],
      "include_companies": ["string"],
      "must_have_certifications": ["string"],
      "languages_required": ["string"]
    }
  },
  "language": "string (ISO 639-1, language of the query itself)"
}
```

### 6.4 Match Result Schema

```json
{
  "match_id": "string (UUID)",
  "query_id": "string (UUID)",
  "profile_id": "string (UUID)",
  "rank": "int",
  "scores": {
    "overall": "float (0.0-1.0)",
    "semantic_similarity": "float (0.0-1.0)",
    "keyword_match": "float (0.0-1.0)",
    "skill_match": "float (0.0-1.0)",
    "experience_match": "float (0.0-1.0)",
    "location_match": "float (0.0-1.0) | null",
    "education_match": "float (0.0-1.0) | null",
    "cross_encoder_score": "float (0.0-1.0)",
    "confidence": "float (0.0-1.0)"
  },
  "rationale": {
    "summary": "string (2-3 sentence overview)",
    "strengths": ["string"],
    "gaps": ["string"],
    "skill_details": [
      {
        "skill": "string",
        "required": "boolean",
        "found": "boolean",
        "proficiency_match": "boolean",
        "evidence": "string"
      }
    ],
    "experience_analysis": "string",
    "recommendation": "strong_match | good_match | potential_match | weak_match"
  },
  "metadata": {
    "search_method": "hybrid | vector_only | keyword_only",
    "reranked": "boolean",
    "language_matched": "boolean (true if cross-lingual match)",
    "passive_candidate": "boolean",
    "processing_time_ms": "int"
  }
}
```

### 6.5 Dataset Requirements for Demo/Evaluation

Since this is a hackathon demo, we need a **synthetic but realistic dataset**. Create:

1. **1,000 candidate profiles** with realistic Indian hiring data:
   - 60% English, 20% Hindi, 10% Tamil, 5% Telugu, 5% other Indian languages
   - Mix of active and passive candidates
   - Varying quality (some complete, some messy)
   - Spread across: software engineering, data science, product, design, marketing

2. **50 job queries** covering:
   - 15 technical roles (DevOps, Backend, ML Engineer, etc.)
   - 15 business roles (Product Manager, Growth, Marketing, etc.)
   - 10 creative roles (Designer, Content, UX, etc.)
   - 10 cross-functional roles (CTO, Head of Product, etc.)

3. **Ground truth labels** for 20 of the 50 queries:
   - Top-10 relevant candidates manually labeled per query
   - Used for evaluation metrics (precision@10, recall@50)

### 6.6 Data Generation Strategy

Use LLM-generated profiles with these requirements:
- Each profile must have at least 3 work experiences
- At least 20% should have non-English names and mixed-language content
- Include realistic Indian company names (Flipkart, Zoho, Freshworks, TCS, Infosys, Razorpay, etc.)
- Include realistic Indian cities (Bangalore, Hyderabad, Pune, Chennai, Noida, etc.)
- Skills should reflect real Indian market demand (Java, Python, React, AWS, Kubernetes, etc.)

---

## 7. Data Models & Schemas

### 7.1 Database Schema (PostgreSQL)

```sql
-- Core profile storage
CREATE TABLE profiles (
    profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(50) NOT NULL,
    raw_text TEXT,
    language_detected VARCHAR(10),
    original_language VARCHAR(10),
    was_translated BOOLEAN DEFAULT FALSE,
    translation_confidence FLOAT,
    data_quality_score FLOAT,
    embedding_vector_id INTEGER,
    bm25_doc_id INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE profile_personal (
    profile_id UUID PRIMARY KEY REFERENCES profiles(profile_id),
    name VARCHAR(255) NOT NULL,
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100) DEFAULT 'India',
    is_remote_ok BOOLEAN DEFAULT FALSE,
    languages_spoken TEXT[],
    native_language VARCHAR(10)
);

CREATE TABLE profile_professional (
    profile_id UUID PRIMARY KEY REFERENCES profiles(profile_id),
    current_title VARCHAR(255),
    current_company VARCHAR(255),
    total_experience_years FLOAT,
    industry VARCHAR(100),
    employment_type VARCHAR(50)
);

CREATE TABLE profile_skills (
    id SERIAL PRIMARY KEY,
    profile_id UUID REFERENCES profiles(profile_id),
    name VARCHAR(255) NOT NULL,
    category VARCHAR(50),
    proficiency VARCHAR(20),
    years_used FLOAT,
    evidence TEXT,
    confidence FLOAT
);

CREATE TABLE profile_experience (
    id SERIAL PRIMARY KEY,
    profile_id UUID REFERENCES profiles(profile_id),
    title VARCHAR(255),
    company VARCHAR(255),
    start_date VARCHAR(10),
    end_date VARCHAR(10),
    is_current BOOLEAN DEFAULT FALSE,
    description TEXT,
    highlights TEXT[],
    skills_demonstrated TEXT[],
    location VARCHAR(255)
);

CREATE TABLE profile_education (
    id SERIAL PRIMARY KEY,
    profile_id UUID REFERENCES profiles(profile_id),
    institution VARCHAR(255),
    degree VARCHAR(255),
    field VARCHAR(255),
    start_date VARCHAR(4),
    end_date VARCHAR(4),
    gpa FLOAT
);

CREATE TABLE profile_signals (
    profile_id UUID PRIMARY KEY REFERENCES profiles(profile_id),
    is_passive BOOLEAN DEFAULT FALSE,
    last_active_date DATE,
    open_to_work BOOLEAN,
    github_activity_score FLOAT,
    has_portfolio BOOLEAN DEFAULT FALSE,
    certifications TEXT[],
    publications TEXT[],
    speaking_engagements TEXT[]
);

-- Query and match storage
CREATE TABLE queries (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    raw_query TEXT NOT NULL,
    parsed JSONB NOT NULL,
    language VARCHAR(10),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE matches (
    match_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES queries(query_id),
    profile_id UUID REFERENCES profiles(profile_id),
    rank INTEGER NOT NULL,
    scores JSONB NOT NULL,
    rationale JSONB NOT NULL,
    search_method VARCHAR(50),
    reranked BOOLEAN DEFAULT FALSE,
    language_matched BOOLEAN DEFAULT FALSE,
    passive_candidate BOOLEAN DEFAULT FALSE,
    processing_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_profiles_language ON profiles(language_detected);
CREATE INDEX idx_profiles_source ON profiles(source);
CREATE INDEX idx_skills_name ON profile_skills(name);
CREATE INDEX idx_skills_profile ON profile_skills(profile_id);
CREATE INDEX idx_experience_profile ON profile_experience(profile_id);
CREATE INDEX idx_matches_query ON matches(query_id);
CREATE INDEX idx_matches_rank ON matches(query_id, rank);
```

---

## 8. Core Features — Functional Requirements

### FR-1: Profile Ingestion & Parsing
- **FR-1.1:** Parse profiles from JSON, CSV, and raw text formats
- **FR-1.2:** Extract structured fields (skills, experience, education) from unstructured text using LLM-assisted parsing
- **FR-1.3:** Normalize all profiles to the unified schema (Section 6.2)
- **FR-1.4:** Assign data quality scores based on completeness (missing fields reduce score)
- **FR-1.5:** Detect original language of the profile
- **FR-1.6:** Generate a `raw_text` concatenation of all fields for embedding

### FR-2: Multilingual Processing
- **FR-2.1:** Detect language of incoming profiles using `langdetect` or `fasttext`
- **FR-2.2:** For non-English profiles, produce both:
  - (a) Preserved original-language text (for embedding)
  - (b) English translation (for keyword search normalization)
- **FR-2.3:** Use `facebook/mbart-large-50-many-to-many-mmt` or `Helsinki-NLP/opus-mt-mul` for translation
- **FR-2.4:** Track translation confidence — flag low-confidence translations for human review
- **FR-2.5:** Use multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) so cross-lingual search works natively
- **FR-2.6:** For code-mixed Hindi-English (Hinglish) text, use HingBERT/HingRoBERTa NER models for entity extraction rather than standard English NER
- **FR-2.7:** Implement Translate-in-Thought (TinT) prompting for the LLM planner to internally translate and process code-mixed queries without explicit translation latency

### FR-3: Hybrid Search Engine (Scoped Retrieval)
- **FR-3.1:** Build a **BM25 index** over all profiles using `rank_bm25` library
- **FR-3.2:** Build a **FAISS vector index** over all profile embeddings
- **FR-3.3:** Apply **structural hard filters BEFORE search** (location, experience range, required certifications) to narrow the candidate pool before vector search — this solves Vector Search Dilution at scale
- **FR-3.4:** Given a query, execute **parallel search** on both indexes (BM25 + FAISS run concurrently)
- **FR-3.5:** Combine results using **Reciprocal Rank Fusion (RRF)**:
  ```
  RRF_score(d) = Σ 1/(k + rank_i(d))  where k = 60
  ```
- **FR-3.6:** Return top-50 candidates from hybrid search for reranking

### FR-4: Cross-Encoder Reranking
- **FR-4.1:** Take top-50 candidates from hybrid search
- **FR-4.2:** For each candidate, create a (query, profile_summary) pair
- **FR-4.3:** Score each pair using a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **FR-4.4:** Re-rank candidates by cross-encoder score
- **FR-4.5:** Return top-10 final candidates with full rationale

### FR-5: Agentic Workflow
- **FR-5.1:** **Planner Agent** parses natural language query into structured search parameters
- **FR-5.2:** **Executor Agent** runs the hybrid search with the parsed parameters
- **FR-5.3:** **Reflector Agent** evaluates each top candidate against the original query:
  - Is this candidate actually a match?
  - What are the gaps?
  - Should we adjust the search?
- **FR-5.4:** If Reflector finds issues, trigger a **re-plan** with adjusted parameters
- **FR-5.5:** Maximum 3 re-plan cycles before returning best available results
- **FR-5.6:** Full agent state (plan, steps, critiques) is logged for debugging

### FR-6: Rationale Generation
- **FR-6.1:** For each top-10 candidate, generate a **Rationale Report** containing:
  - 2-3 sentence summary of why this candidate is a match
  - List of matched skills with evidence snippets
  - List of skill gaps or concerns
  - Experience analysis (relevant vs. irrelevant experience)
  - Overall recommendation: `strong_match | good_match | potential_match | weak_match`
- **FR-6.2:** Rationale must reference specific evidence from the profile (not generic statements)
- **FR-6.3:** Rationale must mention if this was a cross-lingual match (e.g., "This candidate's Hindi resume indicates experience with...")
- **FR-6.4:** Rationale must highlight passive candidate signals if present

### FR-7: Confidence Scoring
- **FR-7.1:** Compute per-dimension confidence scores:
  - `semantic_similarity` (from cosine similarity of embeddings)
  - `keyword_match` (from BM25 score)
  - `skill_match` (weighted overlap of required skills)
  - `experience_match` (years + industry relevance)
  - `location_match` (0.0-1.0, null if not specified)
  - `education_match` (0.0-1.0, null if not specified)
  - `cross_encoder_score` (from reranking)
- **FR-7.2:** Compute `overall` score as weighted combination:
  ```
  overall = 0.25 * semantic + 0.15 * keyword + 0.30 * skill_match 
            + 0.15 * experience + 0.05 * location + 0.05 * education
            + 0.05 * cross_encoder
  ```
- **FR-7.3:** Compute `confidence` based on score variance (high confidence = all dimensions agree)

### FR-8: Listwise Tournament Ranking (Plackett-Luce)
- **FR-8.1:** Instead of independent pointwise scoring, implement a listwise tournament ranking mechanism
- **FR-8.2:** Group top candidates into subsets of 4-5. For each subset, have the evaluator LLM produce a relative ordering
- **FR-8.3:** Aggregate partial rankings from all subsets using a statistical **Plackett-Luce model** to produce a globally coherent final ranking
- **FR-8.4:** This mirrors real-world hiring committees and eliminates score clustering from independent pointwise scoring
- **FR-8.5:** The evaluator LLM judges groups simultaneously, not pairwise (more sample-efficient than pairwise comparisons)

### FR-9: PII Redaction & Bias Masking
- **FR-9.1:** Build an **anonymization layer** that automatically strips PII before candidate data reaches any LLM: names, photos, gendered pronouns, specific local addresses, elite university names, company names
- **FR-9.2:** Implement **style anonymization** — strip LLM-specific stylistic traits (excessive bullet-point structuring, overused verbs like "spearheaded" or "fostered", standard prompt-engineered summaries) before sending to the evaluator LLM
- **FR-9.3:** This prevents **LLM self-preferencing bias** where LLMs rate LLM-written resumes 23-60% higher than human-written ones
- **FR-9.4:** Compute Disparate Impact Ratio after every query; if DIR drops below 0.80 (4/5ths rule), **auto-halt and flag** the results
- **FR-9.5:** Implement `equal_opportunity` metric alongside demographic parity and DIR

---

## 9. API Specifications

### 9.1 POST /api/v1/search

**Request:**
```json
{
  "query": "Find a senior DevOps engineer with 5+ years experience in AWS and Kubernetes, based in Bangalore",
  "filters": {
    "location": "Bangalore",
    "min_experience_years": 5,
    "remote_ok": false
  },
  "max_results": 10,
  "include_rationale": true,
  "language": "en"
}
```

**Response:**
```json
{
  "query_id": "uuid",
  "total_candidates_searched": 1000,
  "results": [
    {
      "rank": 1,
      "profile_id": "uuid",
      "name": "Rahul Sharma",
      "current_title": "Senior DevOps Engineer",
      "current_company": "Razorpay",
      "location": "Bangalore",
      "experience_years": 7,
      "scores": {
        "overall": 0.92,
        "semantic_similarity": 0.88,
        "keyword_match": 0.95,
        "skill_match": 0.94,
        "experience_match": 0.90,
        "location_match": 1.0,
        "confidence": 0.91
      },
      "matched_skills": ["AWS", "Kubernetes", "Docker", "Terraform", "CI/CD"],
      "missing_skills": [],
      "rationale": {
        "summary": "Strong match — Rahul has 7 years of DevOps experience at Razorpay, specifically managing AWS infrastructure and Kubernetes clusters. His experience scaling payment systems aligns well with the seniority and technical requirements.",
        "strengths": [
          "Direct experience with AWS and Kubernetes (3+ years each)",
          "Has scaled infrastructure at a high-traffic fintech company",
          "Based in Bangalore — location match"
        ],
        "gaps": [],
        "recommendation": "strong_match"
      },
      "passive_candidate": false,
      "language_matched": false
    }
  ],
  "processing_time_ms": 1847,
  "search_metadata": {
    "methods_used": ["hybrid", "cross_encoder_rerank"],
    "replan_count": 0,
    "total_time_ms": 1847
  }
}
```

### 9.2 GET /api/v1/profiles/{profile_id}

Returns full normalized profile data.

### 9.3 GET /api/v1/health

Returns system health status including index sizes, model loading status, and last update time.

### 9.4 POST /api/v1/ingest

Bulk ingest profiles from uploaded JSON/CSV file. Returns ingestion report (profiles processed, errors, language distribution).

---

## 10. Agentic Workflow — Detailed Design

### 10.1 State Machine (LangGraph)

```
                    ┌─────────────────┐
                    │   USER QUERY     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    PLANNER      │                     │  (Configurable  │
                    │   LLM Provider) │
                    │ Parse NL query  │
                    │ into structured │
                    │ search params   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    EXECUTOR     │
                    │                 │
                    │ Run hybrid      │
                    │ search with     │
                    │ parsed params   │
                    │                 │
                    │ BM25 + FAISS    │
                    │ → RRF fusion    │
                    │ → Cross-encoder │
                    │   reranking     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   REFLECTOR     │                     │  (Configurable  │
                    │   LLM Provider) │
                    │ For each top-10:│
                    │ - Is it a match?│
                    │ - Gaps? Issues? │
                    │ - Should we     │
                    │   re-search?    │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
            PASS (≥8/10       FAIL (<8/10
            good matches)     good matches)
                    │                 │
                    ▼                 ▼
           ┌──────────┐      ┌──────────┐
           │ GENERATE │      │ RE-PLAN  │
           │ RATIONALE│      │ (max 3x) │
           │ & RETURN │      │          │
           └──────────┘      └──────────┘
                                   │
                                   ▼
                              (back to PLANNER)
```

### 10.2 Agent Prompts

**Planner System Prompt:**
```
You are an expert recruiter's assistant. Given a natural language job query, 
parse it into a structured search specification.

Extract:
- Required skills (with importance: required/preferred/nice_to_have)
- Experience requirements (years, industry)
- Location preferences (city, remote preference)
- Education requirements
- Any exclusion criteria

Output valid JSON matching the Job Query Schema.
If the query is ambiguous, make reasonable assumptions and note them.
```

**Reflector System Prompt:**
```
You are a critical hiring evaluator. For each candidate in the search results, 
assess whether they truly match the job requirements.

For each candidate, provide:
1. overall_assessment: "strong_match" | "good_match" | "potential_match" | "weak_match"
2. key_strengths: list of specific reasons why they match
3. key_gaps: list of specific reasons why they might not match
4. concerns: any red flags or uncertainties
5. should_keep: boolean

Be strict — a "strong_match" means you would confidently shortlist this person.
A "good_match" means they could work with some caveats.
A "potential_match" means they're worth a phone screen.
A "weak_match" means they should be dropped.

Output valid JSON array.
```

---

## 11. Multilingual Processing Pipeline

### 11.1 Language Detection
```python
# Use langdetect for initial detection
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # Reproducibility

def detect_language(text: str) -> dict:
    lang = detect(text)
    return {
        "language": lang,
        "is_english": lang == "en",
        "needs_translation": lang != "en"
    }
```

### 11.2 Translation Pipeline
```python
# For non-English profiles, produce English translation
# Use Helsinki-NLP/opus-mt-mul for multi-pair translation
from transformers import MarianMTModel, MarianTokenizer

def translate_to_english(text: str, source_lang: str) -> dict:
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-en"
    # Fallback to mbart if specific pair not available
    ...
    return {
        "original": text,
        "translated": translated_text,
        "confidence": translation_score
    }
```

### 11.3 Multilingual Embedding
```python
# Use paraphrase-multilingual-MiniLM-L12-v2
# This model maps 50+ languages into the same vector space
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# Output dimension: 384 (use for FAISS: faiss.IndexFlatIP(384))

# Both English and Hindi text of the same meaning → similar vectors
en_embedding = model.encode("Software engineer with Python experience")
hi_embedding = model.encode("पायथन अनुभव वाला सॉफ्टवेयर इंजीनियर")
# cosine_similarity(en_embedding, hi_embedding) ≈ 0.85+
```

### 11.4 Supported Languages (Minimum 10)

| Language | Code | Priority |
|----------|------|----------|
| English | en | P0 |
| Hindi | hi | P0 |
| Tamil | ta | P1 |
| Telugu | te | P1 |
| Marathi | mr | P1 |
| Bengali | bn | P1 |
| Kannada | kn | P1 |
| Malayalam | ml | P2 |
| Gujarati | gu | P2 |
| Punjabi | pa | P2 |
| Odia | or | P3 |
| Assamese | as | P3 |

---

## 12. Ranking & Matching Engine

### 12.1 Skill Matching Algorithm

```python
def compute_skill_match(required_skills: list, candidate_skills: list) -> float:
    """
    Compute skill match score with weighted importance.
    
    Returns: float (0.0 - 1.0)
    """
    total_weight = 0
    matched_weight = 0
    
    for skill in required_skills:
        weight = {
            "required": 1.0,
            "preferred": 0.6,
            "nice_to_have": 0.3
        }[skill["importance"]]
        
        total_weight += weight
        
        # Check if candidate has this skill (with fuzzy matching)
        candidate_match = find_best_match(skill["name"], candidate_skills)
        
        if candidate_match:
            # Partial credit based on proficiency
            proficiency_score = compute_proficiency_match(
                skill.get("min_proficiency"),
                candidate_match.get("proficiency")
            )
            matched_weight += weight * proficiency_score
    
    return matched_weight / total_weight if total_weight > 0 else 0.0


def find_best_match(required_skill: str, candidate_skills: list[dict]) -> dict | None:
    """
    Fuzzy skill matching using:
    1. Exact match
    2. Normalized string match (lowercase, strip)
    3. Embedding similarity (for semantic similarity like "cloud" ≈ "AWS")
    
    Args:
        required_skill: The skill name to match against
        candidate_skills: List of skill dicts from profile schema (each has 'name', 'category', etc.)
    
    Returns:
        The best matching skill dict, or None if no match found.
    """
    # Extract skill names for matching
    skill_names = [s["name"] for s in candidate_skills]
    # ... implementation
```

### 12.2 Experience Matching Algorithm

```python
def compute_experience_match(
    required_min_years: float | None,
    required_max_years: float | None,
    candidate_years: float | None,
    required_industry: str | None,
    candidate_industry: str | None
) -> float:
    """
    Score experience match (0.0 - 1.0).
    """
    score = 0.0
    components = 0
    
    # Years of experience
    if required_min_years and candidate_years is not None:
        components += 1
        if candidate_years >= required_min_years:
            # Bonus for exceeding, diminishing returns
            ratio = min(candidate_years / required_min_years, 2.0)
            score += min(1.0, 0.7 + 0.15 * (ratio - 1.0))
        else:
            # Partial credit if close
            ratio = candidate_years / required_min_years
            score += max(0.0, ratio * 0.7)
    
    # Industry match
    if required_industry and candidate_industry:
        components += 1
        if required_industry.lower() == candidate_industry.lower():
            score += 1.0
        elif semantic_similarity(required_industry, candidate_industry) > 0.7:
            score += 0.6  # Related industry
    
    return score / components if components > 0 else 0.5  # Default neutral
```

### 12.3 Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(
    rankings: list[list[str]],  # Multiple ranked lists of profile_ids
    k: int = 60
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists using RRF.
    
    RRF_score(d) = Σ 1/(k + rank_i(d))
    
    Args:
        rankings: List of ranked profile ID lists
        k: Constant (default 60, per original paper)
    
    Returns:
        Sorted list of (profile_id, rrf_score) tuples
    """
    scores = {}
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 12.4 Listwise Tournament Ranking (Plackett-Luce)

The final ranking stage uses an **Active Listwise Tournament** mechanism:

```python
def rank_via_plackett_luce(
    candidates: list[dict],  # candidate profiles with scores
    evaluator_llm: Callable,  # LLM that judges groups
    group_size: int = 5,
    num_rounds: int = 3,
) -> list[tuple[str, float]]:
    """
    Rank candidates using a Plackett-Luce tournament model.
    
    Process:
    1. Divide candidates into random groups of `group_size`
    2. For each group, LLM produces a relative ordering (listwise, not pairwise)
    3. Aggregate partial rankings using Plackett-Luce into globally coherent ranking
    4. Repeat for `num_rounds` with reshuffled groups
    5. Final ranking from aggregated model
    """
    # 1. Random group assignment
    groups = chunk_randomly(candidates, group_size)
    
    # 2. LLM judges each group
    partial_rankings = []
    for group in groups:
        # Evaluator sees anonymized profiles (PII stripped)
        ordering = evaluator_llm.judge_group(group)
        partial_rankings.append(ordering)
    
    # 3. Aggregate via Plackett-Luce (iterative algorithm)
    # Each candidate's "merit" parameter θ is estimated via MM algorithm
    theta = {c["id"]: 1.0 for c in candidates}
    for _ in range(20):  # EM iterations
        for c in candidates:
            numerator = sum(
                1 for ranking in partial_rankings if c["id"] in ranking
            )
            denominator = sum(
                sum(theta[c2["id"]] for c2 in r if c2["rank"] >= c_rank)
                for r in partial_rankings
                for c_rank in [r.index(c["id"]) + 1 if c["id"] in r else 0]
            )
            if denominator > 0:
                theta[c["id"]] = numerator / denominator
    
    return sorted(candidates, key=lambda c: theta[c["id"]], reverse=True)
```

This approach is superior to:
- **Pointwise scoring** (current): each candidate scored independently, scores cluster and lose discriminative power
- **Pairwise comparisons**: O(n²) comparisons needed; listwise is O(n) while more accurate

---

## 13. Rationale Generation System

### 13.1 Multi-Dimensional YAML Rationale Reports (12-20 Dimensions)

For each candidate in the top-10, generate a **structured YAML report** breaking down evaluation across 12 granular dimensions:

```yaml
candidate_evaluation:
  profile_id: "8f3b92c4-e12a-4c99-b101-0d3e589df462"
  match_confidence: 0.92
  
  dimensional_scores:
    core_technical_skills: 0.95
    tool_proficiency: 0.90
    domain_expertise: 0.85
    role_stability: 0.88
    leadership_indicators: 0.70
    communication_signals: 0.75
    multilingual_fit: 0.90
    localized_salary_alignment: 0.80
    career_growth_trajectory: 0.85
    industry_relevance: 0.92
    company_prestige: 0.78
    culture_fit_signals: 0.82
  
  matching_evidence:
    - dimension: "core_technical_skills"
      skill: "React JS"
      proven_years: 3.5
      context_found: "Lead UI Developer at FinTech Startup"
      proficiency_match: true
    - dimension: "tool_proficiency"
      skill: "Docker"
      proven_years: 2.0
      context_found: "Containerized microservices at Flipkart"
      proficiency_match: true
  
  evaluation_rationale:
    summary: "2-3 sentence overview of why this candidate matches"
    strengths:
      - "Direct experience with React JS in fintech domain"
      - "Based in Bangalore — location match"
    gaps:
      - "No explicit cloud infrastructure experience (AWS/GCP)"
    recommendation: "strong_match"
  
  anonymization_note: "PII stripped before LLM evaluation — no name, university, or location data was visible to the evaluator"
```

### 13.2 Rationale Generation Prompt

```
You are generating a candidate evaluation report for a recruiter.

JOB REQUIREMENTS:
{job_requirements_json}

CANDIDATE PROFILE (PII REDACTED):
{candidate_profile_summary}

MATCH SCORES:
{scores_json}

Generate a detailed rationale report with evaluations across the following 12 dimensions:
1. core_technical_skills — exact match of required programming languages and frameworks
2. tool_proficiency — CI/CD, cloud, monitoring tools
3. domain_expertise — industry-specific knowledge (fintech, ecom, healthcare)
4. role_stability — average tenure, job changes
5. leadership_indicators — team lead, architect, mentoring roles
6. communication_signals — public speaking, technical writing, open source
7. multilingual_fit — language capabilities matching query requirements
8. localized_salary_alignment — compensation parity for their location/seniority
9. career_growth_trajectory — promotions, expanding responsibilities
10. industry_relevance — experience in the same sector as the role
11. company_prestige — brand-name companies vs startups (signal, not bias)
12. culture_fit_signals — open source, side projects, community involvement

Output as structured YAML. Reference specific evidence from the profile.
Do NOT reference names, universities, or locations in the rationale — the candidate was anonymized for bias prevention.
```

---

## 14. Bias Mitigation & Fairness

### 14.1 PII Redaction Layer (Anonymization Pipeline)

Before any candidate data reaches the LLM (for reflection, rationale, or listwise ranking), run through an **anonymization pipeline**:

```python
def anonymize_profile(profile: dict) -> dict:
    """
    Strip all PII from profile before LLM evaluation.
    
    Removes:
    - Candidate name (replace with "Candidate-{uuid}")
    - Gendered pronouns (replace with neutral "they/them")
    - University names (replace with "University-{tier}")
    - Company names (replace with "Company-{size}-{domain}")
    - Specific addresses (replace with city only)
    - Photo URLs
    - Gendered language patterns
    
    Returns: anonymized profile dict (skills, experience lengths, 
             industries, seniority preserved)
    """
    anonymized = deepcopy(profile)
    anonymized["personal"]["name"] = f"Candidate-{uuid4().hex[:8]}"
    anonymized["personal"]["location"]["city"] = None
    anonymized["personal"]["location"]["state"] = None
    anonymized["personal"]["languages_spoken"] = []  # Still tracked for multilingual
    
    for edu in anonymized.get("education", []):
        edu["institution"] = anonymize_institution(edu["institution"])
    
    for exp in anonymized.get("experience", []):
        exp["company"] = anonymize_company(exp["company"])
        exp["location"] = None
    
    return anonymized


def style_anonymize(text: str) -> str:
    """
    Strip LLM-writing style artifacts from profile text.
    
    Removes:
    - Excessive bullet-point structures
    - Overused verbs: "spearheaded", "fostered", "architected", "orchestrated"
    - Standard LLM prompt-engineered summary patterns
    - Generic power phrases
    
    Returns: style-neutralized text
    """
    LLM_VERBS = {"spearheaded", "fostered", "architected", "orchestrated", 
                 "pioneered", "championed", "drove", "delivered", "enabled"}
    # ... replacement logic
    return cleaned_text
```

### 14.2 Internalized Bias Risks

| Risk | Mitigation |
|------|-----------|
| Name-based ethnicity inference | **PII redaction layer** strips names before LLM evaluation. Never infer ethnicity. |
| University pedigree bias | Anonymize institution names to tier-level only before LLM evaluation. Weight skills/experience over education. |
| Gendered language bias | Strip gendered pronouns and style-anonymize text before LLM evaluation. Match on skills only. |
| LLM self-preferencing bias (NEW) | Style-anonymization preprocessor strips LLM-writing artifacts (spearheaded, fostered, architected) before evaluation. Research shows LLMs rate LLM-written resumes 23-60% higher. |
| Location bias (tier-1 vs tier-2) | Anonymize specific location to region only. Do not penalize tier-2 city candidates. |
| Experience bias (gap years) | Do not penalize career gaps. Focus on total relevant experience. |
| Language bias (English-only) | Multilingual support ensures non-English profiles are equally accessible. |

### 14.3 Fairness Metrics + Automated Halting

```python
DISPARATE_IMPACT_THRESHOLD = 0.80  # 4/5ths rule


def compute_fairness_metrics(matches: list, profiles: dict) -> dict:
    """
    Compute fairness metrics for a set of matches.
    """
    return {
        "demographic_parity": compute_demographic_parity(matches, profiles),
        "disparate_impact_ratio": compute_disparate_impact_ratio(matches, profiles),
        "equal_opportunity": compute_equal_opportunity(matches, profiles),
        "language_bias_check": compute_language_bias(matches, profiles),
        "location_bias_check": compute_location_bias(matches, profiles)
    }


def compute_disparate_impact_ratio(matches, profiles):
    """
    4/5ths rule: selection rate of protected group / selection rate of majority group
    Should be ≥ 0.80
    """
    # ... implementation


def check_and_flag_fairness(matches, profiles) -> dict:
    """
    Compute all fairness metrics and flag if DIR < 0.80.
    If DIR drops below threshold, returns a warning that should halt/flag results.
    """
    metrics = compute_fairness_metrics(matches, profiles)
    dir_value = metrics["disparate_impact_ratio"]
    
    if dir_value < DISPARATE_IMPACT_THRESHOLD:
        return {
            "fair": False,
            "warning": f"Disparate Impact Ratio {dir_value:.2f} < {DISPARATE_IMPACT_THRESHOLD}. "
                       f"Results may exhibit bias. Flagged for review.",
            "metrics": metrics,
            "action_required": "review"
        }
    
    return {
        "fair": True,
        "metrics": metrics,
        "action_required": "none"
    }
```

---

## 15. UI/UX Requirements

### 15.1 Demo Application (Gradio)

**Page 1: Search**
- Large search bar with natural language input
- Example queries as clickable chips:
  - "Find a senior Python developer with ML experience in Bangalore"
  - "पायथन और डेटा साइंस में 3 साल का अनुभव वाला उम्मीदवार ढूंढें" (Hindi)
  - "Someone who can build our recommendation engine from scratch"
- Advanced filters (collapsible): Location, Experience, Education, Remote OK
- Loading indicator with search progress

**Page 2: Results**
- Left panel: Ranked candidate cards with:
  - Name, current role, company
  - Overall match score (colored badge)
  - Top 3 matched skills (chips)
  - Quick rationale summary
- Right panel (click a candidate): Full Rationale Report
  - Score breakdown radar chart
  - Detailed skill match table
  - Experience timeline
  - Strengths and gaps lists
  - Recommendation badge

**Page 3: Analytics Dashboard**
- Distribution of matches by language
- Distribution by source
- Average match scores
- Passive vs. active candidate ratio
- Fairness metrics visualization

### 15.2 UI Framework: Gradio (Single Choice)

Use **Gradio** as the sole UI framework. Rationale:
- Simpler API for demos (single-file UI possible)
- Built-in hosting via `gradio deploy` to HuggingFace Spaces (free)
- Native support for markdown, tables, and plots
- No need for separate frontend/backend setup

All UI code lives in `src/ui/app.py`. No Streamlit.

### 15.3 Visual Design Requirements
- Clean, professional look (think: Linear, Vercel dashboard aesthetic)
- Color coding: Green (strong match), Blue (good), Yellow (potential), Red (weak)
- Responsive layout (works on laptop/tablet)
- Dark mode support
- Loading states with skeleton screens
- Smooth transitions between views

---

## 16. Non-Functional Requirements

### 16.1 Performance
| Metric | Target |
|--------|--------|
| Search latency (hybrid search only) | < 500ms |
| End-to-end latency (with rationale) | < 2s |
| Profile ingestion rate | > 100 profiles/second |
| Concurrent searches | ≥ 10 |
| Embedding generation | < 50ms per profile |

### 16.2 Scalability
- System must handle 1,000+ profiles in demo
- Architecture must support scaling to 100K+ profiles (demo shows path to production)
- Vector index should support incremental additions without full rebuild

### 16.3 Reliability
- Graceful degradation: if LLM is unavailable, return matches without rationale
- If cross-encoder is too slow, skip reranking and return hybrid search results
- All errors logged with context for debugging
- System recovers from crashes without data loss (index persistence)

### 16.3a Error Handling Specifications

| Scenario | Fallback Behavior | API Response |
|----------|-------------------|--------------|
| 0 results from search | Return empty results array with message | `{"results": [], "message": "No candidates found matching your criteria. Try broadening your search."}` |
| Translation fails | Use original language profile + multilingual embedding (still works cross-lingually) | Include `"translation_fallback": true` in metadata |
| Profile too noisy to parse | Skip profile, log warning, increment `failed_profiles` counter | Profile excluded from index |
| LLM unavailable (planner) | Fall back to keyword extraction from raw query using spaCy NER | Return results without agentic planning |
| LLM unavailable (rationale) | Return match results with scores but empty `rationale` object | `{"rationale": {"summary": "", "strengths": [], "gaps": [], "recommendation": "good_match"}}` |
| FAISS index corrupted | Rebuild from scratch using stored embeddings in PostgreSQL | System restarts with rebuilt index |
| API rate limit exceeded | Return 429 with retry-after header | `{"error": "Rate limit exceeded", "retry_after_seconds": 60}` |

### 16.3b No-Results Response Schema

```json
{
  "query_id": "uuid",
  "total_candidates_searched": 1000,
  "results": [],
  "message": "No candidates found matching your criteria. Suggestions: broaden location, reduce required experience, or relax skill requirements.",
  "suggestions": [
    "Try removing the location filter",
    "Reduce minimum experience from 5 to 3 years",
    "Move some required skills to preferred"
  ],
  "processing_time_ms": 342
}
```

### 16.4 Security
- No PII exposed in logs (names redacted in debug logs)
- API keys stored in environment variables, never in code
- Input validation on all API endpoints

### 16.5 Observability
- Structured logging (JSON format) for all operations
- Metrics tracking: latency, throughput, error rates
- Search query logging for analysis
- Model loading times tracked

---

## 17. Tech Stack — Pinned Versions

### Core
| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.115+ | API framework |
| Uvicorn | 0.32+ | ASGI server |
| Pydantic | 2.9+ | Data validation |

### ML/NLP
| Technology | Version | Purpose |
|-----------|---------|---------|
| sentence-transformers | 3.3+ | Multilingual embeddings |
| torch | 2.4+ | Backend for transformers |
| transformers | 4.46+ | Translation, cross-encoder |
| langdetect | 1.0.9 | Language detection |
| spacy | 3.8+ | NER, text processing |

### Search
| Technology | Version | Purpose |
|-----------|---------|---------|
| faiss-cpu | 1.8+ | Vector similarity search |
| rank-bm25 | 0.2.2 | BM25 keyword search |
| elastic-transport | (if using Elasticsearch) | Optional: production search |

### Agent Framework
| Technology | Version | Purpose |
|-----------|---------|---------|
| langgraph | 0.2+ | Agentic state machine |
| langchain-core | 0.3+ | LLM abstractions |
| langchain-openai | 0.2+ | OpenAI / OpenAI-compatible endpoints |
| langchain-google-genai | 2.0+ | Google Gemini integration |
| langchain-ollama | 0.2+ | Local Ollama integration |
| openai | 1.52+ | OpenAI SDK (also works with Ollama, vLLM, etc.) |

### Data & Storage
| Technology | Version | Purpose |
|-----------|---------|---------|
| postgresql | 16+ | Profile storage |
| sqlalchemy | 2.0+ | ORM |
| redis | 7+ | Caching (optional) |

### UI
| Technology | Version | Purpose |
|-----------|---------|---------|
| gradio | 5.0+ | Demo UI |
| plotly | 5.24+ | Charts and visualizations |

### DevOps & Testing
| Technology | Version | Purpose |
|-----------|---------|---------|
| pytest | 8.3+ | Testing framework |
| httpx | 0.27+ | API testing |
| ruff | 0.7+ | Linting and formatting |
| mypy | 1.12+ | Type checking |
| docker | 27+ | Containerization |

### Configuration
| Technology | Purpose |
|-----------|---------|
| python-dotenv | Environment variable management |
| toml / pyproject.toml | Project configuration |

---

## 18. Project Structure

```
india-runs/
├── PRD.md                          # This document
├── README.md                       # Project overview and setup
├── pyproject.toml                  # Project config and dependencies
├── Dockerfile                      # Container setup
├── docker-compose.yml              # Local dev environment
│
├── src/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app entry point
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── search.py           # POST /api/v1/search
│   │   │   ├── profiles.py         # GET /api/v1/profiles/{id}
│   │   │   ├── ingest.py           # POST /api/v1/ingest
│   │   │   └── health.py           # GET /api/v1/health
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── logging.py          # Request/response logging
│   │       └── validation.py       # Input validation
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Settings and environment config
│   │   ├── models.py               # Pydantic models (all schemas from Section 7)
│   │   └── constants.py            # Magic numbers, weights, thresholds
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parser.py               # Parse raw profiles into normalized schema
│   │   ├── extractor.py            # LLM-assisted field extraction
│   │   ├── normalizer.py           # Normalize fields across sources
│   │   └── quality_scorer.py       # Data quality scoring
│   │
│   ├── language/
│   │   ├── __init__.py
│   │   ├── detector.py             # Language detection
│   │   ├── translator.py           # Translation pipeline
│   │   └── multilingual.py         # Multilingual embedding utilities
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── hybrid.py               # Hybrid search orchestrator (BM25 + FAISS + RRF)
│   │   ├── vector_search.py        # FAISS vector search wrapper
│   │   ├── bm25_search.py          # BM25 keyword search wrapper
│   │   ├── reranker.py             # Cross-encoder reranking
│   │   └── filters.py              # Hard filter application
│   │
│   ├── matching/
│   │   ├── __init__.py
│   │   ├── skill_matcher.py        # Skill matching with fuzzy logic
│   │   ├── experience_matcher.py   # Experience scoring
│   │   ├── scorer.py               # Overall scoring (weighted combination)
│   │   └── confidence.py           # Confidence calculation
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # LangGraph state machine
│   │   ├── planner.py              # Planner agent
│   │   ├── executor.py             # Executor agent
│   │   ├── reflector.py            # Reflector/critic agent
│   │   └── prompts.py              # All agent system prompts
│   │
│   ├── rationale/
│   │   ├── __init__.py
│   │   ├── generator.py            # Rationale generation (LLM)
│   │   ├── templates.py            # Prompt templates
│   │   └── validator.py            # Validate rationale quality
│   │
│   ├── fairness/
│   │   ├── __init__.py
│   │   ├── bias_detector.py        # Detect potential bias
│   │   └── metrics.py              # Fairness metrics computation
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generator.py            # Synthetic profile generator (for demo)
│   │   ├── ground_truth.py         # Ground truth labels for evaluation
│   │   └── sample_queries.py       # Sample search queries
│   │
│   └── ui/
│       ├── __init__.py
│       ├── app.py                  # Gradio/Streamlit app
│       ├── components.py           # Reusable UI components
│       └── styles.css              # Custom CSS (if Gradio)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_ingestion/
│   │   ├── test_parser.py
│   │   ├── test_extractor.py
│   │   └── test_normalizer.py
│   ├── test_language/
│   │   ├── test_detector.py
│   │   └── test_translator.py
│   ├── test_search/
│   │   ├── test_hybrid.py
│   │   ├── test_vector.py
│   │   ├── test_bm25.py
│   │   └── test_reranker.py
│   ├── test_matching/
│   │   ├── test_skill_matcher.py
│   │   ├── test_scorer.py
│   │   └── test_confidence.py
│   ├── test_agents/
│   │   ├── test_orchestrator.py
│   │   ├── test_planner.py
│   │   └── test_reflector.py
│   ├── test_rationale/
│   │   └── test_generator.py
│   ├── test_api/
│   │   ├── test_search_endpoint.py
│   │   └── test_health_endpoint.py
│   └── test_integration/
│       └── test_end_to_end.py      # Full pipeline integration test
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Explore the synthetic dataset
│   ├── 02_evaluation.ipynb         # Run metrics on test queries
│   └── 03_demo.ipynb               # Interactive demo notebook
│
├── scripts/
│   ├── generate_data.py            # Generate synthetic profiles
│   ├── build_indexes.py            # Build FAISS + BM25 indexes
│   ├── evaluate.py                 # Run evaluation metrics
│   └── deploy.sh                   # Deployment script
│
├── configs/
│   ├── settings.yaml               # Application settings
│   ├── models.yaml                 # Model configurations
│   └── scoring_weights.yaml        # Score weight configuration
│
├── data/
│   ├── profiles/                   # Generated profile data (JSON)
│   ├── queries/                    # Generated queries (JSON)
│   ├── ground_truth/               # Labeled evaluation data
│   ├── indexes/                    # Built FAISS + BM25 indexes
│   └── models/                     # Downloaded model files
│
├── docs/
│   ├── architecture.md             # Architecture deep-dive
│   ├── api.md                      # API documentation
│   ├── evaluation.md               # How to run and interpret metrics
│   └── deployment.md               # Deployment guide
│
└── .env.example                    # Environment variable template
```

---

## 19. Implementation Phases & Timeline

### Phase 1: Bug Fixes & API Wiring (Days 1-3)
**Goal:** Fix all critical bugs from gap analysis — ensure core pipeline works correctly

| Task | Owner | Days | Deliverable |
|------|-------|------|-------------|
| Fix API search route to wire filters into orchestrator | - | 0.5 | `src/api/routes/search.py`, `src/agents/orchestrator.py` |
| Fix `executor.py` to return individual BM25/FAISS scores (not same hybrid score for both) | - | 0.5 | `src/agents/executor.py` |
| Fix `_rationale_node` stub to actually call `RationaleGenerator` | - | 0.5 | `src/agents/orchestrator.py` |
| Fix translation model loading (tokenizer/model swap bug) | - | 0.5 | `src/language/translator.py` |
| Register `InputValidationMiddleware` in `main.py` | - | 0.5 | `src/main.py` |
| Fix health endpoint to reflect actual model state | - | 0.5 | `src/api/routes/health.py` |

**Exit Criteria:** All 7 critical bugs fixed; API filters applied; individual scores correct; rationale generated in agent loop.

### Phase 2: Data Generation (Days 4-6)
**Goal:** Realistic synthetic dataset for demo and evaluation

| Task | Owner | Days | Deliverable |
|------|-------|------|-------------|
| Build synthetic profile generator (1,000 profiles) | - | 2 | `scripts/generate_data.py` |
| Build 50 job queries with Indian-market realism | - | 0.5 | `data/queries/queries.json` |
| Build ground truth labels for 20 queries | - | 0.5 | `data/ground_truth/ground_truth.json` |
| Fix `raw_text` construction to match PRD spec | - | 0.5 | `src/ingestion/normalizer.py` |
| Build FAISS + BM25 indexes from generated data | - | 0.5 | `scripts/build_indexes.py` |

**Exit Criteria:** 1,000 profiles generated, queries and ground truth created, indexes built.

### Phase 3: Scoped Retrieval + Parallel Search (Days 7-9)
**Goal:** Scoped pre-search filtering and parallel BM25/FAISS execution

| Task | Owner | Days | Deliverable |
|------|-------|------|-------------|
| Implement scoped retrieval: structural filters BEFORE BM25/FAISS search | - | 1 | `src/search/filters.py`, `src/agents/executor.py` |
| Make BM25 and FAISS searches run in parallel (concurrent.futures or asyncio.gather) | - | 1 | `src/search/hybrid.py` |
| Add incremental FAISS index support (add vectors without full rebuild) | - | 1 | `src/search/vector_search.py` |

**Exit Criteria:** Filters narrow search pool before vector search; BM25 + FAISS run concurrently; incremental index updates.

### Phase 4: Listwise Tournament Ranking (Days 10-13)
**Goal:** Plackett-Luce tournament ranking replacing pointwise scoring for final ordering

| Task | Owner | Days | Deliverable |
|------|-------|------|-------------|
| Build listwise ranking module (`src/ranking/listwise_ranker.py`) | - | 2 | Plackett-Luce model with group creation + LLM judge calls + MM aggregation |
| Build `judge_group` method for LLM evaluator (judge 4-5 candidates simultaneously) | - | 1 | LLM prompt + structured output parsing |
| Integrate listwise ranking into the agent pipeline (new node after reflect) | - | 1 | `src/agents/orchestrator.py` |
| Unit tests for Plackett-Luce aggregation | - | 1 | `tests/test_matching/test_listwise_ranker.py` |

**Exit Criteria:** Listwise tournament produces globally coherent rankings; outperforms pointwise in precision@10.

### Phase 5: PII Redaction + Bias Automation (Days 14-17)
**Goal:** Anonymization pipeline and automated fairness enforcement

| Task | Owner | Days | Deliverable |
|------|-------|------|-------------|
| Build PII redaction layer (`src/fairness/anonymizer.py`) | - | 1 | Strip names/universities/companies/locations before LLM eval |
| Build style anonymization preprocessor | - | 1 | Strip LLM-writing artifacts (spearheaded, fostered, etc.) |
| Wire anonymization into executor/reflector/rationale pipelines | - | 1 | `src/agents/executor.py`, `src/agents/reflector.py`, `src/rationale/generator.py` |
| Add `equal_opportunity` metric | - | 0.5 | `src/fairness/metrics.py` |
| Add automated DIR check with halting/flagging | - | 0.5 | `src/fairness/anonymizer.py` |
| Unit tests for anonymizer + DIR halting | - | 1 | `tests/test_fairness/` |

**Exit Criteria:** All PII stripped before LLM eval; DIR auto-checks and flags; style-anonymization active.

### Phase 6: 12-20 Dimension Rationale Reports (Days 18-20)
**Goal:** Multi-dimensional YAML rationale with fine-grained evaluation

| Task | Owner | Days | Deliverable |
|------|-------|------|-------------|
| Expand rationale prompt to 12 dimensions | - | 0.5 | `src/rationale/templates.py` |
| Build YAML output formatter for rationale | - | 1 | `src/rationale/generator.py` |
| Update `RationaleValidator` to validate all 12 dimensions | - | 0.5 | `src/rationale/validator.py` |
| Wire rationale into agent's `_rationale_node` | - | 0.5 | `src/agents/orchestrator.py` |
| UI: display 12-dim radar chart + YAML report | - | 0.5 | `src/ui/components.py` |

**Exit Criteria:** Rationale reports contain 12 dimensions; output in YAML; validated by RationaleValidator.

### Phase 7: Code-Mixed NLP + Translation (Days 21-23)
**Goal:** Handle Hindi-English code-mixed profiles and queries

| Task | Owner | Days | Deliverable |
|------|-------|------|-------------|
| Add HingBERT/HingRoBERTa NER for code-mixed entity extraction | - | 1 | `src/language/code_mixed.py` |
| Implement Translate-in-Thought (TinT) prompting for planner | - | 1 | `src/agents/planner.py` |
| Fix translation pipeline to actually load and use translation models | - | 1 | `src/language/translator.py` |
| Cross-lingual evaluation tests | - | 1 | `tests/test_language/test_translator.py` |

**Exit Criteria:** Code-mixed Hindi-English text correctly parsed; translation pipeline functional; cross-lingual MRR ≥ 0.75.

### Phase 8: Error Handling + Observability (Days 24-26)
**Goal:** All 7 PRD fallback scenarios implemented; structured logging

| Task | Owner | Days | Deliverable |
|------|-------|------|-------------|
| Implement 0-results message + suggestions | - | 0.5 | `src/agents/orchestrator.py` |
| Implement translation failure fallback flag | - | 0.5 | `src/language/translator.py` |
| Implement noisy profile skip + counter | - | 0.5 | `src/ingestion/parser.py` |
| Implement FAISS rebuild from stored embeddings | - | 0.5 | `src/search/vector_search.py` |
| Implement rate limiting middleware (429 with retry-after) | - | 0.5 | `src/api/middleware/rate_limit.py` |
| Add structured JSON logging | - | 0.5 | `src/api/middleware/logging.py` |
| Add metrics tracking (latency, throughput, error rates) | - | 0.5 | `src/api/middleware/metrics.py` |

**Exit Criteria:** All 7 error scenarios handled; structured logging active; metrics tracked.

### Phase 9: UI Polish + Full Test Coverage (Days 27-30)
**Goal:** Demo-ready UI, comprehensive test suite, CI pipeline

| Task | Owner | Days | Deliverable |
|------|-------|------|-------------|
| UI: dark mode support | - | 0.5 | `src/ui/styles.css`, `src/ui/app.py` |
| UI: skeleton loading states | - | 0.5 | `src/ui/components.py` |
| UI: live analytics data (not hardcoded) | - | 0.5 | `src/ui/app.py` |
| UI: left panel click→right panel behavior | - | 0.5 | `src/ui/app.py` |
| UI: experience timeline component | - | 0.5 | `src/ui/components.py` |
| Missing tests: 11 missing test files | - | 2 | `tests/test_ingestion/test_extractor.py`, etc. |
| Missing tests: critical test cases (cross-lingual, listwise, latency) | - | 1 | All test files |
| Set up CI pipeline (.github/workflows/) | - | 0.5 | `.github/workflows/test.yml` |
| Set up Gradio Spaces config | - | 0.5 | `gradio_deploy.py` or Spaces config |

**Exit Criteria:** Dark mode, skeleton loading, live analytics; 100+ tests; CI passing.

### Phase 10: Documentation + Submission (Days 31-33)
**Goal:** Complete documentation, pitch deck, deployable demo

| Task | Owner | Days | Deliverable |
|------|-------|------|-------------|
| Update README.md with new architecture and metrics | - | 0.5 | `README.md` |
| Update docs/ with new features (listwise, anonymizer, etc.) | - | 1 | `docs/architecture.md`, `docs/api.md`, etc. |
| Create pitch deck PDF (12 slides) | - | 1 | `docs/pitch_deck.pdf` |
| Fix Dockerfile to build indexes + expose Gradio | - | 0.5 | `Dockerfile`, `docker-compose.yml` |
| Deploy to HuggingFace Spaces | - | 0.5 | Live demo URL |
| Final commit + push to GitHub | - | 0.5 | GitHub repo |

**Exit Criteria:** Fully working demo deployed; pitch deck ready; GitHub repo public.

---

## 20. Testing Strategy

### 20.1 Test Types

| Type | Framework | Coverage Target | When |
|------|-----------|----------------|------|
| Unit Tests | pytest | ≥ 80% | Every phase |
| Integration Tests | pytest + httpx | Key workflows | End of each phase |
| Evaluation Tests | Custom scripts | Precision@10, Recall@50 | Phase 2, 5 |
| E2E Tests | pytest | Full pipeline | Phase 4, 6 |
| Fairness Tests | pytest + custom | Disparate impact, PII redaction | Phase 5 |
| Load Tests | Locust or custom | 10 concurrent searches | Phase 6 |
| Listwise Ranking Tests | pytest | Plackett-Luce correctness | Phase 4 |
| Anonymization Tests | pytest | PII stripping, style stripping | Phase 5 |

### 20.2 Critical Test Cases

```python
# tests/test_search/test_hybrid.py

def test_hybrid_search_returns_ranked_results():
    """Hybrid search returns ranked results with scores > 0"""

def test_rrf_fusion_combines_bm25_and_vector():
    """RRF correctly merges results from both search methods"""

def test_scoped_retrieval_filters_before_search():
    """Structural filters narrow the candidate pool BEFORE vector search runs"""

def test_bm25_and_faiss_run_in_parallel():
    """BM25 and FAISS searches execute concurrently, not sequentially"""

def test_cross_lingual_search_matches_hindi_to_english():
    """A Hindi query matches English profiles with similar content"""

def test_multilingual_embedding_same_space():
    """English and Hindi versions of same text have cosine similarity > 0.8"""

def test_reranker_improves_precision():
    """Cross-encoder reranking improves precision@10 vs raw hybrid"""

# tests/test_matching/test_skill_matcher.py

def test_exact_skill_match():
    """Python matches Python exactly"""

def test_fuzzy_skill_match():
    """'React.js' matches 'React'"""

def test_semantic_skill_match():
    """'Cloud computing' matches 'AWS'"""

def test_required_skill_missing_penalizes():
    """Missing required skill heavily penalizes score"""

def test_nice_to_have_skill_bonuses():
    """Nice-to-have skill adds small bonus"""

# tests/test_agents/test_orchestrator.py

def test_full_pipeline_end_to_end():
    """Full pipeline: query → plan → search → reflect → rationale"""

def test_replan_triggered_on_poor_results():
    """Reflector triggers re-plan when < 6/10 matches are good"""

def test_max_replan_limit():
    """System stops after 3 re-plan cycles even if results are poor"""

def test_rationale_node_generates_rationale():
    """_rationale_node calls RationaleGenerator and returns structured output"""

# tests/test_integration/test_end_to_end.py

def test_search_with_multilingual_profiles():
    """Search 100 profiles (20 Hindi, 80 English) with Hindi query"""

def test_search_with_messy_profiles():
    """Search works on profiles with missing fields and noisy data"""

def test_search_latency_under_2s():
    """End-to-end search with rationale completes in < 2 seconds"""

# NEW: Phase 4 - Listwise Ranking

def test_plackett_luce_aggregation():
    """Plackett-Luce correctly aggregates partial rankings into global ordering"""

def test_listwise_ranked_better_than_pointwise():
    """Listwise ranking produces better precision@10 than pointwise scoring"""

# NEW: Phase 5 - Anonymization

def test_pii_stripped_before_llm():
    """Names, universities, and companies are removed before LLM evaluation"""

def test_style_anonymization_removes_llm_artifacts():
    """LLM-writing style artifacts (spearheaded, fostered) are stripped"""

def test_dir_below_threshold_flags_results():
    """DIR < 0.80 triggers a flag/warning in the response"""

def test_equal_opportunity_metric_computed():
    """Equal opportunity metric is computed alongside demographic parity"""

# NEW: Phase 7 - Code-Mixed

def test_hingbert_ner_extracts_skills_from_hinglish():
    """HingBERT correctly extracts 'Python' and 'Docker' from 'Mujhe Python aur Docker aata hai'"""

def test_translate_in_thought_handles_code_mixed_query():
    """TinT prompting correctly parses a code-mixed Hindi-English query"""

# NEW: Phase 8 - Error Handling

def test_zero_results_returns_message_with_suggestions():
    """Empty results include a helpful message and broadening suggestions"""

def test_translation_fallback_sets_flag():
    """When translation fails, translation_fallback=true is set in metadata"""

def test_rate_limiting_returns_429():
    """Exceeding rate limit returns 429 with retry-after header"""

def test_faiss_corruption_rebuilds_from_embeddings():
    """Corrupted FAISS index triggers automatic rebuild from stored embeddings"""
```

### 20.3 Evaluation Protocol

```python
# scripts/evaluate.py

def evaluate_search_system(test_queries, ground_truth, search_fn):
    """
    Run full evaluation on test set.
    
    Metrics:
    - precision@k for k in [1, 3, 5, 10]
    - recall@k for k in [10, 25, 50]
    - MRR (Mean Reciprocal Rank)
    - nDCG@10
    - Cross-lingual MRR (subset of queries in non-English)
    - Latency p50, p95, p99
    """
    results = {
        "precision": {k: [] for k in [1, 3, 5, 10]},
        "recall": {k: [] for k in [10, 25, 50]},
        "mrr": [],
        "ndcg": [],
        "cross_lingual_mrr": [],
        "latencies": []
    }
    
    for query, relevant_ids in ground_truth.items():
        start = time.time()
        matches = search_fn(query)
        latency = (time.time() - start) * 1000
        
        results["latencies"].append(latency)
        # ... compute all metrics
    
    # Report
    print(f"Precision@10: {mean(results['precision'][10]):.3f}")
    print(f"Recall@50: {mean(results['recall'][50]):.3f}")
    print(f"MRR: {mean(results['mrr']):.3f}")
    print(f"Latency p50: {percentile(results['latencies'], 50):.0f}ms")
    print(f"Latency p95: {percentile(results['latencies'], 95):.0f}ms")
```

---

## 21. Deployment Requirements

### 21.1 Local Development
```bash
# Quick start
docker-compose up -d          # Start PostgreSQL + Redis
python scripts/generate_data.py   # Generate synthetic data
python scripts/build_indexes.py   # Build FAISS + BM25 indexes
uvicorn src.main:app --reload     # Start API server
python src/ui/app.py              # Start Gradio UI
```

### 21.2 Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install -e .
COPY . .
RUN python scripts/build_indexes.py
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 21.3 Demo Deployment Options
1. **Gradio Live** — Deploy via `gradio deploy` to HuggingFace Spaces (free)
2. **Railway/Render** — Deploy Docker container (free tier available)
3. **Local** — Run on laptop, present via screen share

---

## 22. Submission Package

### 22.1 Required Deliverables

1. **Pitch Deck (PDF)** containing:
   - Problem statement (with the DevOps keyword example)
   - Solution overview (architecture diagram)
   - Live demo screenshots or GIFs
   - Metrics (precision@10, latency, cross-lingual performance)
   - Innovation highlights (agentic workflow, rationale reports)
   - Impact statement (passive talent, multilingual, fairness)
   - Team info and contact

2. **Working Demo** accessible via URL or local run

3. **Source Code** on GitHub with:
   - Clear README.md
   - Setup instructions
   - Test suite passing
   - Architecture documentation

### 22.2 Pitch Deck Structure (10-12 slides)

| Slide | Content |
|-------|---------|
| 1 | Title: "Intelligent Candidate Discovery — Beyond Keywords" |
| 2 | Problem: The keyword matching failure (DevOps example) |
| 3 | Problem: Indian market challenges (multilingual, messy data, passive talent) |
| 4 | Solution: High-level architecture diagram |
| 5 | Innovation: Agentic workflow (Plan → Execute → Reflect) |
| 6 | Innovation: Hybrid search (BM25 + Vector + Cross-Encoder) |
| 7 | Innovation: Multilingual support (30+ languages) |
| 8 | Demo: Live search + Rationale report screenshot |
| 9 | Metrics: Precision, Recall, Latency, Cross-lingual MRR |
| 10 | Impact: Passive talent discovery, fairness, accessibility |
| 11 | Roadmap: Path to production (scale to 700M profiles) |
| 12 | Thank you + Contact |

---

## 23. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLM API costs exceed budget | Medium | High | Use GPT-4o-mini or local Ollama (free), cache responses, limit to planning + rationale only |
| Cross-encoder too slow | Medium | Medium | Use MiniLM variant, cache, skip if latency > 500ms |
| Translation quality too low | Medium | Medium | Use multilingual embeddings as primary, translation as secondary |
| Synthetic data not realistic enough | Medium | Medium | Use real company names, Indian cities, mixed skill levels |
| Demo fails during presentation | Low | Critical | Pre-compute top results, have offline fallback, test extensively |
| FAISS index too large for free tier | Low | Medium | Use 384-dim embeddings (MiniLM), quantize if needed |
| Agent hallucinates bad search parameters | Medium | Medium | Validate planner output against schema, use structured output |
| LLM self-preferencing bias inflates scores | Medium | High | Style-anonymization preprocessor strips LLM-writing artifacts from profiles before evaluation |
| Plackett-Luce model converges slowly | Low | Medium | Cap EM iterations at 20; fall back to pointwise if convergence fails |
| PII redaction removes too much context | Medium | Medium | Selective redaction: preserve skills, years, industries; strip names, institutions, addresses |

---

## 24. Appendix

### A. Glossary

| Term | Definition |
|------|-----------|
| **BM25** | Best Matching 25 — a probabilistic retrieval function that scores documents based on term frequency and inverse document frequency |
| **FAISS** | Facebook AI Similarity Search — a library for efficient similarity search of dense vectors |
| **RRF** | Reciprocal Rank Fusion — a method to combine multiple ranked lists into a single ranking |
| **Cross-Encoder** | A model that takes (query, document) pair and outputs a relevance score — more accurate but slower than bi-encoders |
| **Bi-Encoder** | A model that encodes query and document independently into vectors — fast but less accurate |
| **Agentic Workflow** | An AI system that uses multiple agents (planner, executor, reflector) to solve complex tasks iteratively |
| **Passive Candidate** | A professional who is not actively looking for a job but may be open to the right opportunity |
| **Plackett-Luce** | A statistical model for ranking items based on partial preference observations (tournament ranking) |
| **Listwise Ranking** | A ranking approach where candidates are judged in groups simultaneously, not independently |
| **Scoped Retrieval** | Applying structural filters before vector search to narrow the candidate pool (solves Vector Search Dilution) |
| **PII Redaction** | Personally Identifiable Information removal — stripping names, universities, companies before LLM evaluation |
| **Style Anonymization** | Removing LLM-writing style artifacts from candidate text to prevent LLM self-preferencing bias |
| **HingBERT** | A BERT model fine-tuned on Hindi-English code-mixed text for NER and classification |
| **Translate-in-Thought (TinT)** | A prompting strategy where the LLM internally translates code-mixed text without explicit translation calls |

### B. Reference Implementations

1. **Hybrid Search with RRF:** [LangChain Hybrid RAG](https://python.langchain.com/docs/how_to/hybrid_search/)
2. **Agentic Plan-Execute-Reflect:** [LangGraph Plan-and-Execute](https://langchain-ai.github.io/langgraph/tutorials/reflection/reflection/)
3. **Multilingual Embeddings:** [Sentence-Transformers Multilingual](https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models)
4. **Cross-Encoder Reranking:** [Sentence-Transformers Cross-Encoder](https://www.sbert.net/docs/package/cross_encoder/cross_encoder.html)

### C. Evaluation Benchmarks

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| Precision@10 | Of top-10 returned, how many are relevant? | `relevant_in_top_10 / 10` |
| Recall@50 | Of all relevant candidates, how many are in top-50? | `relevant_in_top_50 / total_relevant` |
| MRR | How high is the first relevant result? | `1 / rank_of_first_relevant` |
| nDCG@10 | Are the best matches ranked highest? | Normalized Discounted Cumulative Gain |
| Cross-lingual MRR | Does cross-lingual search work? | MRR on non-English queries only |

### D. Scoring Weights (Tunable)

**Source of truth:** `configs/scoring_weights.yaml`. The code in `src/matching/scorer.py` loads weights from this file at runtime. Do NOT hardcode weights in Python — always read from YAML.

**Note on `cross_encoder_score`:** This score is only available for candidates that made it through reranking. For candidates not reranked (e.g., cross-encoder skipped due to latency), this component defaults to 0.5 (neutral).

**Note on listwise ranking:** After pointwise scoring and Plackett-Luce tournament ranking, the listwise rank is the final ordering. The pointwise scores remain as interpretable dimensions.

```yaml
# configs/scoring_weights.yaml
# These weights must sum to 1.0
scoring_weights:
  semantic_similarity: 0.25   # Cosine similarity from multilingual embeddings
  keyword_match: 0.15         # BM25 score (normalized)
  skill_match: 0.30           # Weighted skill overlap (see FR-7.2)
  experience_match: 0.15      # Years + industry relevance
  location_match: 0.05        # 0.0-1.0, null if not specified in query
  education_match: 0.05       # 0.0-1.0, null if not specified in query
  cross_encoder: 0.05         # Cross-encoder score (see note above)

skill_importance_weights:
  required: 1.0
  preferred: 0.6
  nice_to_have: 0.3

proficiency_scores:
  beginner: 0.25
  intermediate: 0.50
  advanced: 0.75
  expert: 1.00

rrf_k: 60  # From original RRF paper (Cormack et al. 2009), tunable

max_replan_cycles: 3
min_good_matches_for_pass: 8

# Fairness thresholds
fairness:
  disparate_impact_threshold: 0.80  # 4/5ths rule
  auto_flag_on_violation: true

# Listwise ranking
listwise_ranking:
  enabled: true
  group_size: 5
  max_em_iterations: 20
  num_tournament_rounds: 3
```

---

**END OF PRD**

*This document is the single source of truth for building the Intelligent Candidate Discovery system. All implementation decisions should trace back to requirements in this document.*
