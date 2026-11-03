from __future__ import annotations

from src.core.query_parser import detect_multi_word_skills, expand_with_aliases, parse_query


def _names(skills) -> list[str]:
    return [s.name.lower() for s in skills]


def test_parse_query_extracts_skills_city_and_experience():
    parsed = parse_query("Senior Python developer in Bangalore with 5+ years experience")
    assert "python" in _names(parsed.required_skills)
    assert parsed.location.city == "Bangalore"
    assert parsed.experience.min_years == 5.0
    assert parsed.location.remote_ok is False
    assert parsed.original_query.startswith("Senior Python")


def test_parse_query_experience_range():
    parsed = parse_query("django engineer 3-6 years")
    assert parsed.experience.min_years == 3.0
    assert parsed.experience.max_years == 6.0


def test_parse_query_remote_flag():
    assert parse_query("remote react developer").location.remote_ok is True


def test_parse_query_filler_words_are_not_skills():
    parsed = parse_query("find a strong senior engineer for the team with python")
    names = _names(parsed.required_skills)
    for filler in ("find", "strong", "senior", "engineer", "team"):
        assert filler not in names


def test_parse_query_no_skills_falls_back_to_original_text():
    parsed = parse_query("someone great")
    assert len(parsed.required_skills) == 1
    assert parsed.required_skills[0].name == "someone great"


def test_parse_query_dedupes_skills():
    parsed = parse_query("python python python docker")
    names = _names(parsed.required_skills)
    assert names.count("python") == 1


def test_parse_query_overflow_goes_to_preferred():
    q = "python java rust go kotlin swift scala ruby php dart"
    parsed = parse_query(q)
    assert len(parsed.required_skills) == 8
    assert len(parsed.preferred_skills) >= 1


def test_detect_multi_word_skills_respects_word_boundaries():
    found = [s.lower() for s in detect_multi_word_skills("expert in machine learning and nlp")]
    assert "machine learning" in found
    assert detect_multi_word_skills("xmachine learningx") == []


def test_expand_with_aliases_keeps_original_first():
    expansions = expand_with_aliases("python developer")
    assert expansions[0] == "python developer"
    assert len(expansions) >= 1
