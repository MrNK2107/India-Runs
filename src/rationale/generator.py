from __future__ import annotations

import json
import logging
from typing import Any

from src.core.config import get_llm_client, get_settings
from src.core.models import (
    MatchRecommendation,
    MatchResult,
    Profile,
    Rationale,
    SkillDetail,
)

logger = logging.getLogger(__name__)


class RationaleGenerator:
    def __init__(self) -> None:
        self._client = None
        settings = get_settings()
        self.model = settings.openai_model

    @property
    def client(self):
        if self._client is None:
            try:
                self._client = get_llm_client()
            except Exception:
                logger.warning("LLM client unavailable for rationale generator")
                self._client = None
        return self._client

    async def generate(
        self, match: MatchResult, profile: Profile, job_requirements: dict,
    ) -> Rationale:
        try:
            if self.client is None:
                raise RuntimeError("LLM client unavailable")

            prompt = self._build_prompt(match, profile, job_requirements)
            from langchain_core.messages import HumanMessage

            messages = [
                HumanMessage(
                    content="Generate a candidate evaluation report for a recruiter. "
                    f"Output valid JSON only.\n\n{prompt}"
                ),
            ]
            response = await self.client.ainvoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            return self._parse_response(content)
        except Exception as e:
            logger.warning(f"Rationale LLM failed, using template: {e}")
            return self._template_rationale(match, profile)

    async def generate_batch(
        self, matches: list[MatchResult], profiles: dict[str, Profile],
        job_requirements: dict[str, Any],
    ) -> list[Rationale]:
        results: list[Rationale] = []
        for m in matches:
            profile = profiles.get(m.profile_id)
            if profile is None:
                continue
            rationale = await self.generate(m, profile, job_requirements)
            results.append(rationale)
        return results

    def _build_prompt(
        self, match: MatchResult, profile: Profile, job_requirements: dict,
    ) -> str:
        skills_text = ", ".join(s.name for s in profile.skills)
        return (
            f"Candidate: {match.name}\n"
            f"Title: {match.current_title or 'N/A'}\n"
            f"Company: {match.current_company or 'N/A'}\n"
            f"Experience: {match.experience_years or 0} years\n"
            f"Skills: {skills_text}\n"
            f"Location: {match.location or 'N/A'}\n"
            f"Overall Score: {match.scores.overall:.2f}\n"
            f"Skill Match: {match.scores.skill_match:.2f}\n"
            f"Experience Match: {match.scores.experience_match:.2f}\n"
            f"Matched Skills: {', '.join(match.matched_skills)}\n"
            f"Missing Skills: {', '.join(match.missing_skills)}\n"
        )

    def _parse_response(self, response: str) -> Rationale:
        try:
            data = json.loads(response)
            return Rationale(
                summary=data.get("summary", ""),
                strengths=data.get("strengths", []),
                gaps=data.get("gaps", []),
                skill_details=[
                    SkillDetail(**sd) for sd in data.get("skill_details", [])
                ],
                experience_analysis=data.get("experience_analysis", ""),
                recommendation=MatchRecommendation(data.get("recommendation", "good_match")),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return Rationale(summary="Could not parse rationale response from LLM.")

    def _template_rationale(self, match: MatchResult, profile: Profile | None) -> Rationale:
        matched = set(match.matched_skills)
        missing = set(match.missing_skills)
        skill_details = [
            SkillDetail(
                skill=s,
                required=True,
                found=s in matched,
                proficiency_match=s in matched,
                evidence=s in matched and "Found in profile" or "Not found in profile",
            )
            for s in list(matched) + list(missing)
        ]

        strengths = []
        if matched:
            top5 = ", ".join(list(matched)[:5])
            strengths.append(f"Matches {len(matched)} required skills: {top5}")
        if match.experience_years and match.scores.experience_match >= 0.6:
            yrs = match.experience_years
            strengths.append(f"Experience level ({yrs} years) meets requirements")
        if match.current_company:
            strengths.append(f"Currently at {match.current_company}")

        gaps = []
        if missing:
            gaps.append(f"Missing {len(missing)} required skills: {', '.join(list(missing)[:5])}")
        if match.scores.experience_match < 0.5 and match.experience_years:
            gaps.append(f"Experience ({match.experience_years} years) may be insufficient")

        if match.scores.overall >= 0.7:
            recommendation = MatchRecommendation.STRONG
        elif match.scores.overall >= 0.5:
            recommendation = MatchRecommendation.GOOD
        elif match.scores.overall >= 0.3:
            recommendation = MatchRecommendation.POTENTIAL
        else:
            recommendation = MatchRecommendation.WEAK

        return Rationale(
            summary=(
                f"{match.name} is a{'n' if recommendation == MatchRecommendation.STRONG else ''} "
                f"{recommendation.value.replace('_', ' ')} "
                f"with {len(matched)} matching skills "
                f"and {match.experience_years or 0} years of experience."
            ),
            strengths=strengths,
            gaps=gaps,
            skill_details=skill_details,
            experience_analysis=(
                f"Candidate has {match.experience_years or 0} years of experience"
                f"{' at ' + match.current_company if match.current_company else ''}."
            ),
            recommendation=recommendation,
        )
