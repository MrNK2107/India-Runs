from __future__ import annotations

import json
import logging
import re

from src.agents.prompts import PLANNER_SYSTEM_PROMPT
from src.core.config import get_llm_client, get_settings
from src.core.constants import INDIAN_CITIES, INDIAN_COMPANIES
from src.core.models import ParsedQuery, PreferredSkill, RequiredSkill, SkillImportance
from src.language.code_mixed import CodeMixedProcessor
from src.matching.skill_matcher import SKILL_ALIASES

logger = logging.getLogger(__name__)


class PlannerAgent:
    def __init__(self) -> None:
        self.client = get_llm_client()
        settings = get_settings()
        self.model = settings.openai_model

    async def plan(self, raw_query: str) -> ParsedQuery:
        try:
            processor = CodeMixedProcessor()
            if processor.detect_code_mixed(raw_query):
                logger.info("Code-mixed query detected, applying TinT prompting")
                tint_query = (
                    "[Translate-in-Thought] The following query contains Hinglish "
                    "(Hindi-English code-mixed text). Internally translate it to English "
                    "before parsing, then output the JSON result.\n\n"
                    "Query: " + raw_query
                )
            else:
                tint_query = raw_query

            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=tint_query),
            ]
            response = await self.client.ainvoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = json.loads(content)
            return ParsedQuery(**parsed)
        except Exception as e:
            logger.warning(f"Planner LLM failed, using fallback: {e}")
            return self._fallback_parse(raw_query)

    async def replan(
        self, original_query: str, previous_params: dict, feedback: str,
    ) -> ParsedQuery:
        try:
            prompt = (
                "Original query: " + original_query + "\n"
                "Previous params: " + json.dumps(previous_params) + "\n"
                "Feedback: " + feedback + "\n"
                "Revise the search parameters. Output valid JSON only."
            )
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = await self.client.ainvoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = json.loads(content)
            return ParsedQuery(**parsed)
        except Exception as e:
            logger.warning(f"Replan LLM failed, using fallback: {e}")
            relaxed = self._relax_params(previous_params)
            return ParsedQuery(**relaxed)

    def _fallback_parse(self, query: str) -> ParsedQuery:
        required: list[RequiredSkill] = []
        preferred: list[PreferredSkill] = []
        min_years: float | None = None
        max_years: float | None = None
        industry: str | None = None
        city: str | None = None

        for alias, aliases in SKILL_ALIASES.items():
            candidates = [alias] + aliases
            if any(qs in query.lower() for qs in candidates):
                required.append(
                    RequiredSkill(name=alias.title(), importance=SkillImportance.REQUIRED)
                )

        year_match = re.search(r"(\d+)\+?\s*(?:years?|yrs?)", query.lower())
        if year_match:
            min_years = float(year_match.group(1))

        year_range = re.search(r"(\d+)\s*[-to]+\s*(\d+)\s*(?:years?|yrs?)", query.lower())
        if year_range:
            min_years = float(year_range.group(1))
            max_years = float(year_range.group(2))

        for c in INDIAN_CITIES:
            if c.lower() in query.lower():
                city = c
                break

        for comp in INDIAN_COMPANIES:
            if comp.lower() in query.lower():
                industry = "technology"
                break

        return ParsedQuery(
            required_skills=required,
            preferred_skills=preferred,
            experience={"min_years": min_years, "max_years": max_years, "industry": industry},
            location={"city": city, "remote_ok": "remote" in query.lower()},
        )

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
