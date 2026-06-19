from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.normalizer import normalize_redrob as new_normalizer

_PROFESSIONAL_FIELDS = [
    "current_title",
    "current_company",
    "total_experience_years",
    "industry",
]

_SAMPLE_ROWS = [
    {
        "candidate_id": "CAND_001",
        "profile": {
            "anonymized_name": "Ira Vora",
            "current_title": "Backend Engineer",
            "current_company": "Mindtree",
            "years_of_experience": 6.9,
            "current_industry": "it_services",
            "location": "Toronto, Ontario",
            "country": "Canada",
            "headline": "Backend Engineer | SQL, Spark, Cloud",
            "summary": "Software/data professional with 6.9 yrs experience",
        },
        "career_history": [
            {"title": "Backend Engineer", "company": "Mindtree",
             "start_date": "2024-03-08", "is_current": True,
             "description": "Implemented streaming data pipelines"},
            {"title": "Data Engineer", "company": "OldCo",
             "start_date": "2021-01", "end_date": "2024-02",
             "description": "Built ETL pipelines"},
        ],
        "education": [
            {"institution": "LPU", "degree": "B.E.",
             "field_of_study": "CS", "end_year": 2020, "grade": "8.24 CGPA"},
        ],
        "skills": [
            {"name": "NLP", "proficiency": "advanced", "endorsements": 37, "duration_months": 26},
            {"name": "AWS", "proficiency": "beginner", "endorsements": 5, "duration_months": 8},
        ],
        "languages": [{"language": "English", "proficiency": "professional"}],
    },
    {
        "candidate_id": "CAND_002",
        "profile": {
            "anonymized_name": "Sparse Candidate",
            "headline": "Senior Python Dev",
        },
        "career_history": [],
        "skills": [],
    },
    {
        "candidate_id": "CAND_003",
        "profile": {
            "anonymized_name": "Minimal",
        },
        "career_history": [],
        "skills": [],
    },
]


class TestRegressionNoDataLoss:
    def test_all_non_none_fields_preserved(self):
        for row in _SAMPLE_ROWS:
            profile = new_normalizer(row)
            for field in _PROFESSIONAL_FIELDS:
                old_val = row.get("profile", {}).get({
                    "current_title": "current_title",
                    "current_company": "current_company",
                    "total_experience_years": "years_of_experience",
                    "industry": "current_industry",
                }[field])
                new_val = getattr(profile.professional, field)
                if old_val is not None:
                    assert new_val is not None, (
                        f"{row['candidate_id']}: {field} was '{old_val}' "
                        f"in raw data but new normalizer returned None"
                    )

    def test_seniority_level_is_int_when_found(self):
        profile = new_normalizer(_SAMPLE_ROWS[0])
        assert isinstance(profile.professional.seniority_level, int)

    def test_seniority_level_can_be_none(self):
        profile = new_normalizer(_SAMPLE_ROWS[2])
        assert profile.professional.seniority_level is None

    def test_normalizer_never_crashes(self):
        for row in _SAMPLE_ROWS:
            profile = new_normalizer(row)
            assert profile.profile_id is not None
