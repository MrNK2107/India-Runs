from __future__ import annotations

from src.extraction.domain import extract_industry


class TestExtractIndustry:
    def test_direct(self):
        val, src = extract_industry(
            {"current_industry": "fintech"}, [], [],
        )
        assert val == "fintech"
        assert src == "direct"

    def test_from_company_map(self):
        val, src = extract_industry(
            {},
            [],
            [{"company": "Razorpay"}],
        )
        assert val == "fintech"
        assert src == "company_map"

    def test_from_skills(self):
        val, src = extract_industry(
            {},
            [{"name": "NLP"}, {"name": "PyTorch"}, {"name": "Computer Vision"}],
            [],
        )
        assert val == "ai/ml"
        assert src == "skills"

    def test_from_headline(self):
        val, src = extract_industry(
            {"headline": "Fintech Engineer at a payments startup"},
            [],
            [],
        )
        assert val == "fintech"
        assert src == "headline_summary"

    def test_no_info_returns_none(self):
        val, src = extract_industry({}, [], [])
        assert val is None
        assert src == "not_found"
