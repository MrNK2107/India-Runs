from __future__ import annotations

from src.extraction.company import extract_company


class TestExtractCompany:
    def test_direct(self):
        val, src = extract_company(
            {"current_company": "Mindtree"}, [],
        )
        assert val == "Mindtree"
        assert src == "direct"

    def test_fallback_to_history(self):
        val, src = extract_company(
            {},
            [{"company": "Mindtree", "start_date": "2020-01", "is_current": True}],
        )
        assert val == "Mindtree"
        assert src == "history"

    def test_prefers_current_role(self):
        val, src = extract_company(
            {"current_company": "Mindtree"},
            [{"company": "OldCo", "start_date": "2020-01", "is_current": False}],
        )
        assert val == "Mindtree"
        assert src == "direct"

    def test_no_info_returns_none(self):
        val, src = extract_company({}, [])
        assert val is None
        assert src == "not_found"
