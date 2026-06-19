from __future__ import annotations

from src.core.models import RequiredSkill, Skill, SkillImportance
from src.matching.skill_matcher import SkillMatcher


def test_skill_matcher_exact(sample_profile):
    matcher = SkillMatcher()
    required = [RequiredSkill(name="Python", importance=SkillImportance.REQUIRED)]
    score, details = matcher.match_skills(required, sample_profile.skills)
    assert score > 0
    assert details[0]["found"]


def test_skill_matcher_missing(sample_profile):
    matcher = SkillMatcher()
    required = [RequiredSkill(name="Rust", importance=SkillImportance.REQUIRED)]
    score, details = matcher.match_skills(required, sample_profile.skills)
    assert score == 0
    assert not details[0]["found"]


def test_skill_matcher_alias(sample_profile):
    matcher = SkillMatcher()
    required = [RequiredSkill(name="AWS", importance=SkillImportance.REQUIRED)]
    score, details = matcher.match_skills(required, sample_profile.skills)
    assert score > 0


def test_skill_matcher_fuzzy():
    matcher = SkillMatcher(similarity_threshold=0.5)
    skills = [Skill(name="Python3", category="programming_language")]
    required = [RequiredSkill(name="Python", importance=SkillImportance.REQUIRED)]
    score, details = matcher.match_skills(required, skills)
    assert score > 0


def test_skill_matcher_empty_required(sample_profile):
    matcher = SkillMatcher()
    score, details = matcher.match_skills([], sample_profile.skills)
    assert score == 1.0


def test_importance_weights():
    matcher = SkillMatcher()
    from src.core.models import SkillImportance
    assert matcher._importance_weight(SkillImportance.REQUIRED) == 1.0
    assert matcher._importance_weight(SkillImportance.PREFERRED) == 0.6


def test_experience_matcher():
    from src.matching.experience_matcher import ExperienceMatcher
    em = ExperienceMatcher()
    score = em.match(required_min_years=3, candidate_years=5)
    assert score > 0.5


def test_experience_matcher_underqualified():
    from src.matching.experience_matcher import ExperienceMatcher
    em = ExperienceMatcher()
    score = em.match(required_min_years=5, candidate_years=1)
    assert score < 0.5


def test_scorer_overall(sample_profile):
    from src.core.models import MatchScores
    from src.matching.scorer import CandidateScorer
    scorer = CandidateScorer()
    scores = scorer.compute_overall({
        "semantic_similarity": 0.8,
        "keyword_match": 0.7,
        "skill_match": 0.9,
        "experience_match": 0.6,
        "location_match": 1.0,
        "education_match": 0.5,
        "cross_encoder_score": 0.75,
    })
    assert isinstance(scores, MatchScores)
    assert 0.0 <= scores.overall <= 1.0
    assert scores.overall > 0.5


def test_scorer_confidence():
    from src.matching.confidence import compute_confidence
    conf = compute_confidence({"a": 0.8, "b": 0.7, "c": 0.9})
    assert 0.0 <= conf <= 1.0


def test_confidence_module():
    from src.matching.confidence import compute_confidence, compute_score_variance
    var = compute_score_variance([0.8, 0.7, 0.9])
    assert var >= 0
    conf = compute_confidence({"a": 0.8, "b": 0.7, "c": 0.9})
    assert 0.0 <= conf <= 1.0
