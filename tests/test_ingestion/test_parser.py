from __future__ import annotations

import json

import pytest

from src.ingestion.normalizer import normalize_redrob
from src.ingestion.parser import ProfileParser


@pytest.fixture
def parser():
    return ProfileParser()


def test_parse_jsonl(tmp_path, parser):
    lines = [
        json.dumps({
            "candidate_id": str(i),
            "profile": {"anonymized_name": f"User {i}"},
        })
        for i in range(3)
    ]
    file_path = tmp_path / "test.jsonl"
    file_path.write_text("\n".join(lines))
    results = list(parser.parse_jsonl_file(file_path))
    assert len(results) == 3


def test_parse_json(tmp_path, parser):
    data = [{"candidate_id": "1"}, {"candidate_id": "2"}]
    file_path = tmp_path / "test.json"
    file_path.write_text(json.dumps(data))
    results = parser.parse_json_file(file_path)
    assert len(results) == 2


def test_parse_json_single_object(tmp_path, parser):
    data = {"candidate_id": "1"}
    file_path = tmp_path / "test.json"
    file_path.write_text(json.dumps(data))
    results = parser.parse_json_file(file_path)
    assert len(results) == 1


def test_parse_empty_file(tmp_path, parser):
    file_path = tmp_path / "empty.jsonl"
    file_path.write_text("")
    results = list(parser.parse_jsonl_file(file_path))
    assert len(results) == 0


def test_normalizer_redrob():
    raw = {
        "candidate_id": "redrob-1",
        "profile": {
            "anonymized_name": "Priya Sharma",
            "current_title": "Software Engineer",
            "current_company": "Flipkart",
            "location": "Bangalore",
            "years_of_experience": 5.0,
        },
        "skills": [
            {"name": "Python", "proficiency": "advanced"},
            {"name": "SQL", "proficiency": "intermediate"},
        ],
    }
    profile = normalize_redrob(raw)
    assert profile.profile_id == "redrob-1"
    assert profile.personal.name == "Priya Sharma"
    assert profile.professional.current_title == "Software Engineer"
    assert profile.professional.current_company == "Flipkart"
    assert profile.personal.location.city == "Bangalore"
    assert len(profile.skills) == 2


def test_normalizer_redrob_minimal():
    raw = {"candidate_id": "r2", "profile": {"anonymized_name": "Raj"}}
    profile = normalize_redrob(raw)
    assert profile.profile_id == "r2"
    assert profile.personal.name == "Raj"
    assert profile.professional.total_experience_years is None


def test_quality_scorer(sample_profile):
    from src.ingestion.quality_scorer import compute_data_quality_score
    score = compute_data_quality_score(sample_profile)
    assert 0.0 <= score <= 1.0
    assert score > 0.5
