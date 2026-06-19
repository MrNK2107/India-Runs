from __future__ import annotations

from src.extraction.pipeline import FieldExtractorPipeline


class TestPipeline:
    def test_full_profile(self):
        raw = {
            "candidate_id": "CAND_001",
            "profile": {
                "anonymized_name": "Test User",
                "current_title": "Senior Backend Engineer",
                "current_company": "Mindtree",
                "years_of_experience": 6.5,
                "current_industry": "it_services",
                "location": "Bangalore, Karnataka",
                "country": "India",
                "headline": "Senior Backend Engineer | Mindtree",
            },
            "career_history": [
                {"title": "Backend Engineer", "company": "Mindtree",
                 "start_date": "2024-01", "is_current": True},
            ],
            "skills": [
                {"name": "Python", "proficiency": "advanced"},
                {"name": "AWS", "proficiency": "intermediate"},
            ],
        }
        bundle = FieldExtractorPipeline().extract(raw)
        assert bundle.current_title.value == "Senior Backend Engineer"
        assert bundle.current_title.source == "direct"
        assert bundle.current_company.value == "Mindtree"
        assert bundle.current_company.source == "direct"
        assert bundle.total_experience_years.value is not None
        assert bundle.industry.value == "it_services"
        assert bundle.industry.source == "direct"
        assert bundle.seniority_level.value == 3

    def test_sparse_profile_falls_back_gracefully(self):
        raw = {
            "candidate_id": "CAND_002",
            "profile": {
                "anonymized_name": "Sparse User",
                "headline": "Python Developer | 5+ yrs experience",
            },
            "career_history": [],
            "skills": [],
        }
        bundle = FieldExtractorPipeline().extract(raw)
        assert bundle.current_title.value is not None
        assert bundle.current_title.source == "headline"
        assert bundle.current_company.value is None
        assert bundle.total_experience_years.value == 5.0
        assert bundle.total_experience_years.source == "regex_fallback"
        assert bundle.industry.value is None
        assert bundle.seniority_level.value is not None

    def test_empty_raw(self):
        bundle = FieldExtractorPipeline().extract({})
        assert bundle.current_title.value is None
        assert bundle.current_company.value is None
        assert bundle.total_experience_years.value is None
        assert bundle.industry.value is None
        assert bundle.seniority_level.value is None
