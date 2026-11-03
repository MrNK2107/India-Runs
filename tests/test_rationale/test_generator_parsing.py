from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from src.core.models import (
    MatchMetadata,
    MatchRecommendation,
    MatchResult,
    MatchScores,
    PersonalInfo,
    Profile,
    SearchMethod,
)
from src.rationale.generator import RationaleGenerator


def _match(overall: float, matched: list[str], missing: list[str]) -> MatchResult:
    return MatchResult(
        query_id="q", profile_id="p", rank=1, name="Asha",
        scores=MatchScores(overall=overall, skill_match=0.5, experience_match=0.7),
        matched_skills=matched, missing_skills=missing, experience_years=6,
        metadata=MatchMetadata(search_method=SearchMethod.HYBRID),
    )



@pytest.mark.parametrize("overall,phrase", [
    (0.9, "is a strong match"), (0.6, "is a good match"),
    (0.4, "is a potential match"), (0.1, "is a weak match"),
])
def test_template_summary_grammar(overall: float, phrase: str) -> None:
    # Regression: strong matches read "Asha is an strong match".
    r = RationaleGenerator()._template_rationale(_match(overall, ["Go"], []), None)
    assert phrase in r.summary


def test_template_keeps_skill_order_and_dedupes() -> None:
    m = _match(0.8, ["Python", "Django", "AWS", "Python"], ["Rust", "Go"])
    r = RationaleGenerator()._template_rationale(m, None)
    assert [d.skill for d in r.skill_details] == ["Python", "Django", "AWS", "Rust", "Go"]
    assert "Python, Django, AWS" in r.strengths[0]
    assert "Rust, Go" in r.gaps[0]


def test_template_text_is_identical_across_processes() -> None:
    # Skills went through set(), whose order depends on per-process string hashing.
    code = (
        "from src.core.models import *;from src.rationale.generator import RationaleGenerator;"
        "m=MatchResult(query_id='q',profile_id='p',rank=1,name='A',scores=MatchScores(overall=.8),"
        "matched_skills=['a','b','c','d','e','f','g'],missing_skills=['x','y','z'],"
        "metadata=MatchMetadata(search_method=SearchMethod.HYBRID));"
        "r=RationaleGenerator()._template_rationale(m,None);print(r.strengths[0]+'|'+r.gaps[0])"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
            env={**__import__("os").environ, "PYTHONHASHSEED": str(seed)},
        ).stdout
        for seed in (1, 2, 3)
    }
    assert len(outs) == 1


class FakeLLM:
    def __init__(self, content: Any) -> None:
        self.content = content

    async def ainvoke(self, messages: Any) -> Any:
        return SimpleNamespace(content=self.content)


def _gen(content: Any) -> RationaleGenerator:
    g = RationaleGenerator()
    g._client = FakeLLM(content)
    return g


GOOD = {
    "summary": "Solid backend engineer.", "strengths": ["Go"], "gaps": [],
    "skill_details": [], "experience_analysis": "6 years", "recommendation": "strong_match",
}


async def test_generate_parses_plain_and_fenced_json() -> None:
    profile = Profile(personal=PersonalInfo(name="Asha"))
    m = _match(0.8, ["Go"], [])
    for content in (json.dumps(GOOD), f"```json\n{json.dumps(GOOD)}\n```"):
        r = await _gen(content).generate(m, profile, {})
        assert r.summary == "Solid backend engineer."
        assert r.recommendation == MatchRecommendation.STRONG


@pytest.mark.parametrize("content", [
    "not json", "[]", json.dumps({**GOOD, "recommendation": "amazing"}),
    json.dumps({**GOOD, "skill_details": [{"skill": "Go"}]}),
])
async def test_unusable_llm_output_falls_back_to_template(content: str) -> None:
    # Previously returned a near-empty "Could not parse rationale response" rationale.
    profile = Profile(personal=PersonalInfo(name="Asha"))
    r = await _gen(content).generate(_match(0.8, ["Go"], ["Rust"]), profile, {})
    assert "Could not parse" not in r.summary
    assert "is a strong match" in r.summary
    assert r.gaps
