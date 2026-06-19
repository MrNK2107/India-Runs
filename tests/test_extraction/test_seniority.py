from __future__ import annotations

from src.extraction.seniority import extract_seniority


class TestExtractSeniority:
    def test_intern(self):
        val, src = extract_seniority("Software Intern", None, [])
        assert val == 0

    def test_junior_title(self):
        val, src = extract_seniority("Junior Engineer", None, [])
        assert val == 1

    def test_senior(self):
        val, src = extract_seniority("Senior Engineer", None, [])
        assert val == 3

    def test_principal(self):
        val, src = extract_seniority("Principal Architect", None, [])
        assert val == 5

    def test_cto(self):
        val, src = extract_seniority("Chief Technology Officer", None, [])
        assert val == 6

    def test_years_fallback_junior(self):
        val, src = extract_seniority(None, 1, [])
        assert val == 1
        assert src == "years_fallback"

    def test_years_fallback_mid(self):
        val, src = extract_seniority(None, 3, [])
        assert val == 2
        assert src == "years_fallback"

    def test_years_fallback_senior(self):
        val, src = extract_seniority(None, 6, [])
        assert val == 3
        assert src == "years_fallback"

    def test_years_fallback_lead(self):
        val, src = extract_seniority(None, 10, [])
        assert val == 4
        assert src == "years_fallback"

    def test_years_fallback_principal(self):
        val, src = extract_seniority(None, 15, [])
        assert val == 5
        assert src == "years_fallback"

    def test_no_info_returns_none(self):
        val, src = extract_seniority(None, None, [])
        assert val is None
        assert src == "not_found"

    def test_domain_override_professor(self):
        val, src = extract_seniority("Professor of Computer Science", None, [])
        assert val == 6
        assert src == "domain_override"

    def test_prefers_title_over_years(self):
        val, src = extract_seniority("Junior Dev", 10, [])
        assert val == 1
        assert src == "title_keyword"

    def test_fallback_to_history_title(self):
        history = [
            {"title": "Senior Engineer", "start_date": "2020-01", "is_current": True},
        ]
        val, src = extract_seniority(None, None, history)
        assert val == 3
        assert src == "history_title"
