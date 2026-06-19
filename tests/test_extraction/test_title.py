from __future__ import annotations

from src.extraction.title import extract_title


class TestExtractTitle:
    def test_direct(self):
        val, src = extract_title(
            {"current_title": "Backend Engineer"}, [],
        )
        assert val == "Backend Engineer"
        assert src == "direct"

    def test_direct_empty_string_ignored(self):
        val, src = extract_title(
            {"current_title": ""},
            [{"title": "Senior Dev", "start_date": "2020-01", "is_current": True}],
        )
        assert val == "Senior Dev"
        assert src == "history"

    def test_fallback_to_history(self):
        val, src = extract_title(
            {},
            [{"title": "Senior Backend Engineer", "start_date": "2020-01", "is_current": True}],
        )
        assert val == "Senior Backend Engineer"
        assert src == "history"

    def test_fallback_to_headline_with_pipe(self):
        val, src = extract_title(
            {"headline": "Senior Python Backend | Acme Corp"}, [],
        )
        assert val is not None
        assert src == "headline"

    def test_headline_no_pipe(self):
        val, src = extract_title(
            {"headline": "Senior Software Engineer"}, [],
        )
        assert src == "headline"

    def test_no_info_returns_none(self):
        val, src = extract_title({}, [])
        assert val is None
        assert src == "not_found"

    def test_prefers_direct_over_history(self):
        val, src = extract_title(
            {"current_title": "Lead Engineer"},
            [{"title": "Junior Dev", "start_date": "2020-01", "is_current": True}],
        )
        assert val == "Lead Engineer"
        assert src == "direct"
