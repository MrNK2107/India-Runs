from __future__ import annotations

from src.core.models import (
    MatchMetadata,
    MatchRecommendation,
    MatchResult,
    MatchScores,
    PersonalInfo,
    Profile,
    Rationale,
    SearchMethod,
)
from src.rationale.generator import RationaleGenerator
from src.rationale.validator import RationaleValidator


def test_rationale_validator_valid():
    validator = RationaleValidator()
    rationale = Rationale(
        summary="Good match with strong skills.",
        strengths=["Python expert"],
        gaps=[],
        recommendation=MatchRecommendation.GOOD,
    )
    is_valid, issues = validator.validate(rationale)
    assert is_valid
    assert len(issues) == 0


def test_rationale_validator_invalid():
    validator = RationaleValidator()
    rationale = Rationale(
        summary="Short",
        strengths=[],
        gaps=[],
        recommendation=MatchRecommendation.GOOD,
    )
    is_valid, issues = validator.validate(rationale)
    assert not is_valid
    assert len(issues) > 0


def test_rationale_validator_batch():
    validator = RationaleValidator()
    rationales = [
        Rationale(summary="Good match", strengths=["Python"],
                  recommendation=MatchRecommendation.STRONG),
        Rationale(summary="X", strengths=[], recommendation=MatchRecommendation.GOOD),
    ]
    stats = validator.validate_batch(rationales)
    assert stats["total"] == 2
    assert stats["invalid"] == 1


def test_template_rationale():
    generator = RationaleGenerator()
    match = MatchResult(
        query_id="q1",
        profile_id="p1",
        rank=1,
        name="Test User",
        scores=MatchScores(
            overall=0.85, semantic_similarity=0.8, keyword_match=0.7,
            skill_match=0.9, experience_match=0.6, confidence=0.8,
        ),
        matched_skills=["Python", "Django"],
        missing_skills=["Rust"],
        metadata=MatchMetadata(search_method=SearchMethod.HYBRID),
    )
    profile = Profile(personal=PersonalInfo(name="Test User"))
    rationale = generator._template_rationale(match, profile)
    assert isinstance(rationale, Rationale)
    assert len(rationale.strengths) > 0
