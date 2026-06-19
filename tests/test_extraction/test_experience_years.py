from __future__ import annotations

from src.extraction.experience_years import extract_experience_years


class TestExtractExperienceYears:
    def test_direct_field(self):
        val, src = extract_experience_years(
            {"years_of_experience": 5.0}, [],
        )
        assert val == 5.0
        assert src == "direct_structured"

    def test_dates_from_history(self):
        val, src = extract_experience_years(
            {},
            [
                {"start_date": "2020-01", "end_date": "2022-01", "is_current": False},
                {"start_date": "2022-02", "end_date": "2024-02", "is_current": False},
            ],
        )
        assert src == "dates_structured"
        assert val is not None and 3.9 < val < 4.1

    def test_average_when_close(self):
        val, src = extract_experience_years(
            {"years_of_experience": 4.0},
            [
                {"start_date": "2020-06", "end_date": "2024-06", "is_current": False},
            ],
        )
        assert src == "average"
        assert val is not None and 3.9 < val < 4.1

    def test_dates_preferred_when_disagree_and_confident(self):
        val, src = extract_experience_years(
            {"years_of_experience": 10.0},
            [
                {"start_date": "2023-01", "end_date": "2024-01", "is_current": False},
            ],
        )
        assert src == "dates_structured"
        assert val is not None and 0.9 < val < 1.1

    def test_regex_fallback_from_headline(self):
        val, src = extract_experience_years(
            {"headline": "Senior Dev with 8+ years experience"},
            [],
        )
        assert val == 8.0
        assert src == "regex_fallback"

    def test_regex_fallback_from_summary(self):
        val, src = extract_experience_years(
            {"summary": "Experience: 12 years in software engineering"},
            [],
        )
        assert val == 12.0
        assert src == "regex_fallback"

    def test_typo_100_plus_years_ignored(self):
        val, src = extract_experience_years(
            {"headline": "100+ years of experience"},
            [],
        )
        assert val is None
        assert src == "not_found"

    def test_typo_150_years_ignored(self):
        val, src = extract_experience_years(
            {"years_of_experience": 150},
            [],
        )
        assert val is None
        assert src == "not_found"

    def test_no_info_returns_none(self):
        val, src = extract_experience_years({}, [])
        assert val is None
        assert src == "not_found"
