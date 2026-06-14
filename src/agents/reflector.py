from __future__ import annotations

import json
import logging

from src.agents.prompts import REFLECTOR_SYSTEM_PROMPT
from src.core.config import get_llm_client, get_settings
from src.core.models import MatchResult, ParsedQuery

logger = logging.getLogger(__name__)


class ReflectorAgent:
    def __init__(self) -> None:
        self.client = get_llm_client()
        settings = get_settings()
        self.model = settings.openai_model

    async def reflect(self, query: ParsedQuery, results: list[MatchResult]) -> dict:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            candidates_text = json.dumps(
                [
                    {
                        "profile_id": r.profile_id,
                        "name": r.name,
                        "title": r.current_title,
                        "company": r.current_company,
                        "overall_score": r.scores.overall,
                        "confidence": r.scores.confidence,
                        "matched_skills": r.matched_skills[:10],
                        "missing_skills": r.missing_skills[:10],
                    }
                    for r in results[:20]
                ],
                indent=2,
            )

            messages = [
                SystemMessage(content=REFLECTOR_SYSTEM_PROMPT),
                HumanMessage(content=f"Candidates:\n{candidates_text}"),
            ]
            response = await self.client.ainvoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            evaluations = json.loads(content)

        except Exception as e:
            logger.warning(f"Reflector LLM failed, using fallback: {e}")
            evaluations = self._fallback_evaluate(results)

        should_replan = self._should_replan(evaluations)
        good_matches = {"strong_match", "good_match"}
        good_count = sum(
            1 for ev in (evaluations if isinstance(evaluations, list) else [])
            if isinstance(ev, dict) and ev.get("overall_assessment") in good_matches
        )

        return {
            "evaluations": evaluations if isinstance(evaluations, list) else [],
            "good_match_count": good_count,
            "should_replan": should_replan,
            "feedback": self._generate_feedback(evaluations, results),
        }

    def _fallback_evaluate(self, results: list[MatchResult]) -> list[dict]:
        evaluations: list[dict] = []
        for r in results[:20]:
            if r.scores.overall >= 0.8:
                assessment = "strong_match"
            elif r.scores.overall >= 0.6:
                assessment = "good_match"
            elif r.scores.overall >= 0.4:
                assessment = "potential_match"
            else:
                assessment = "weak_match"

            evaluations.append({
                "profile_id": r.profile_id,
                "overall_assessment": assessment,
                "key_strengths": r.matched_skills[:5],
                "key_gaps": r.missing_skills[:5],
                "concerns": [],
                "should_keep": assessment in ("strong_match", "good_match", "potential_match"),
            })
        return evaluations

    def _should_replan(self, evaluations: list | dict, threshold: int = 8) -> bool:
        if isinstance(evaluations, list):
            good_matches = {"strong_match", "good_match"}
            good = sum(
                1 for ev in evaluations
                if isinstance(ev, dict) and ev.get("overall_assessment") in good_matches
            )
            return good < threshold
        return False

    def _generate_feedback(self, evaluations: list | dict, results: list[MatchResult]) -> str:
        if isinstance(evaluations, list):
            weak = [
                ev for ev in evaluations
                if isinstance(ev, dict) and ev.get("overall_assessment") == "weak_match"
            ]
            if weak:
                return f"{len(weak)} weak matches found. Consider broadening criteria."
        if not results:
            return "No results found. Query may be too restrictive."
        return "Results look reasonable."

    def _relax_params(self, params: dict) -> dict:
        params = dict(params)
        exp = dict(params.get("experience", {}))
        if exp.get("min_years") is not None:
            exp["min_years"] = max(0, exp["min_years"] - 2)
        if exp.get("max_years") is not None:
            exp["max_years"] = (exp["max_years"] or 0) + 3
        params["experience"] = exp

        loc = dict(params.get("location", {}))
        loc["city"] = None
        loc["remote_ok"] = True
        params["location"] = loc

        params["required_skills"] = []
        return params
