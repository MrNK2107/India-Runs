from __future__ import annotations

import pytest

from src.agents.planner import PlannerAgent


@pytest.mark.asyncio
async def test_planner_fallback():
    planner = PlannerAgent()
    result = await planner.plan("Find a Python developer in Bangalore with 3 years experience")
    parsed = result
    assert hasattr(parsed, "required_skills")
    assert len(parsed.required_skills) > 0


@pytest.mark.asyncio
async def test_planner_fallback_skills():
    planner = PlannerAgent()
    result = await planner.plan("Need a senior DevOps engineer with AWS and Kubernetes")
    skill_names = [rs.name.lower() for rs in result.required_skills]
    assert any("aws" in s for s in skill_names)


@pytest.mark.asyncio
async def test_planner_fallback_years():
    planner = PlannerAgent()
    result = await planner.plan("Need a candidate with 5+ years of experience")
    assert result.experience.min_years == 5


@pytest.mark.asyncio
async def test_planner_fallback_city():
    planner = PlannerAgent()
    result = await planner.plan("Looking for someone in Mumbai")
    assert result.location.city == "Mumbai"


@pytest.mark.asyncio
async def test_planner_relax_params():
    planner = PlannerAgent()
    params = {
        "experience": {"min_years": 5, "max_years": 10},
        "location": {"city": "Bangalore", "remote_ok": False},
        "required_skills": [{"name": "Python", "importance": "required"}],
    }
    relaxed = planner._relax_params(params)
    assert relaxed["experience"]["min_years"] == 3
    assert relaxed["location"]["city"] is None
    assert relaxed["location"]["remote_ok"] is True


@pytest.mark.asyncio
async def test_reflector_fallback():
    from src.agents.reflector import ReflectorAgent
    from src.core.models import MatchMetadata, MatchResult, MatchScores, ParsedQuery, SearchMethod
    reflector = ReflectorAgent()
    match = MatchResult(
        query_id="q1", profile_id="p1", rank=1, name="User",
        scores=MatchScores(overall=0.85, semantic_similarity=0.8, keyword_match=0.7,
                           skill_match=0.9, experience_match=0.6, confidence=0.8),
        metadata=MatchMetadata(search_method=SearchMethod.HYBRID),
    )
    result = await reflector.reflect(ParsedQuery(), [match])
    assert "evaluations" in result
    assert "should_replan" in result


@pytest.mark.asyncio
async def test_reflector_should_replan_below_threshold():
    from src.agents.reflector import ReflectorAgent
    reflector = ReflectorAgent()
    evals = [
        {"overall_assessment": "strong_match", "should_keep": True},
        {"overall_assessment": "weak_match", "should_keep": False},
    ]
    assert reflector._should_replan(evals, threshold=2)
    assert not reflector._should_replan(evals, threshold=1)
