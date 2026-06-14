# API Documentation — India Runs

Base URL: `http://localhost:8000/api/v1`

## Endpoints

### `GET /health`

Returns system health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "index_size": 100000,
  "models_loaded": {
    "embedding": false,
    "cross_encoder": false
  }
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/health
```

---

### `POST /search`

Search for candidates matching a job description.

**Request Body:**
```json
{
  "query": "Senior Python developer with experience in Django and AWS",
  "filters": {
    "min_experience_years": 3,
    "location": "Bangalore"
  },
  "max_results": 10
}
```

**Response:**
```json
{
  "query_id": "q-001",
  "query": "Senior Python developer...",
  "results": [
    {
      "profile_id": "redrob-42",
      "rank": 1,
      "name": "Priya Sharma",
      "scores": {
        "overall": 0.87,
        "semantic_similarity": 0.82,
        "keyword_match": 0.75,
        "skill_match": 0.90,
        "experience_match": 0.85,
        "confidence": 0.78
      },
      "matched_skills": ["Python", "Django", "SQL"],
      "missing_skills": ["AWS"],
      "rationale": {
        "summary": "Strong match with 5 years Python experience",
        "strengths": ["Python advanced", "Django expertise"],
        "gaps": ["Missing AWS experience"],
        "recommendation": "Strong"
      },
      "metadata": {
        "search_method": "hybrid",
        "processing_time_ms": 145
      }
    }
  ],
  "total_results": 1,
  "search_metadata": {
    "total_processing_time_ms": 320,
    "search_method": "hybrid",
    "replan_cycles_used": 0
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Python developer Bangalore", "max_results": 5}'
```

---

### `GET /profiles/{profile_id}`

Retrieve a specific profile by ID.

**Example:**
```bash
curl http://localhost:8000/api/v1/profiles/redrob-42
```

**Response:** Full Profile object with personal info, professional history, skills, education, experience, and signals.

---

### `GET /profiles`

List profiles with pagination.

**Query Parameters:**
- `skip` (int, default: 0) — Number of profiles to skip
- `limit` (int, default: 20) — Maximum number of profiles to return

**Example:**
```bash
curl "http://localhost:8000/api/v1/profiles?skip=0&limit=10"
```

---

### `POST /ingest`

Upload a JSON file containing profiles. Accepts both single objects and arrays.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` — JSON file upload

**Response:**
```json
{
  "total_profiles": 5,
  "successful": 5,
  "failed": 0,
  "errors": []
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@profiles.json"
```

## Interactive Docs

When running locally, visit http://localhost:8000/docs for the Swagger UI.
