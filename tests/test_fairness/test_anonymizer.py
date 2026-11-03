from __future__ import annotations

from src.core.models import Education, PersonalInfo, Profile, WorkExperience
from src.fairness.anonymizer import (
    anonymize_profile,
    anonymize_text_for_bias,
    style_anonymize,
)


def test_anonymize_profile_strips_pii(sample_profile):
    result = anonymize_profile(sample_profile)
    flat = repr(result)
    assert "Test User" not in flat
    assert "IIT Bombay" not in flat
    assert "Bangalore" not in flat
    assert result["anonymized_id"].startswith("Candidate-")
    assert result["skills"] == ["Python", "Django", "AWS"]
    assert result["skill_count"] == 3
    assert result["experience_years"] == 5.0
    assert result["education_tier"] == "tier-1"
    assert result["location_tier"] == "metro"
    assert result["has_degree"] is True


def test_anonymized_id_is_deterministic(sample_profile):
    assert (
        anonymize_profile(sample_profile)["anonymized_id"]
        == anonymize_profile(sample_profile)["anonymized_id"]
    )


def test_education_and_location_tiers_for_sparse_profile():
    p = Profile(profile_id="x", personal=PersonalInfo(name="A"))
    result = anonymize_profile(p)
    assert result["education_tier"] == "unknown"
    assert result["location_tier"] == "unknown"
    assert result["has_degree"] is False
    assert result["experience_years"] is None


def test_tier2_education_and_location_from_experience():
    p = Profile(
        profile_id="y",
        personal=PersonalInfo(name="B"),
        education=[Education(institution="Some State College")],
        experience=[WorkExperience(title="Dev", company="C", location="Indore, MP")],
    )
    result = anonymize_profile(p)
    assert result["education_tier"] == "tier-2"
    assert result["location_tier"] == "tier-2/3"


def test_style_anonymize_replaces_llm_verbs_and_phrases():
    out = style_anonymize("Spearheaded a team. Passionate about   leveraged data.")
    assert "Spearheaded" not in out
    assert "led" in out
    assert "used" in out
    assert "passionate about" not in out.lower()
    assert "  " not in out


def test_anonymize_text_for_bias_redacts_names_and_universities():
    out = anonymize_text_for_bias("Ravi Kumar studied at IIT Madras")
    assert "Ravi Kumar" not in out
    assert "[NAME]" in out
    assert "[UNIVERSITY]" in out
