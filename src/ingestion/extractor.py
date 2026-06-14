from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class FieldExtractor:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from src.core.config import get_llm_client

            self._client = get_llm_client()
        return self._client

    async def extract_skills(self, text: str) -> list[dict[str, Any]]:
        prompt = self._build_extraction_prompt(text, "skills")
        response = await self._call_llm(prompt)
        return self._parse_llm_response(response, "skills")

    async def extract_experience(self, text: str) -> list[dict[str, Any]]:
        prompt = self._build_extraction_prompt(text, "experience")
        response = await self._call_llm(prompt)
        return self._parse_llm_response(response, "experience")

    async def extract_education(self, text: str) -> list[dict[str, Any]]:
        prompt = self._build_extraction_prompt(text, "education")
        response = await self._call_llm(prompt)
        return self._parse_llm_response(response, "education")

    async def extract_all(self, text: str) -> dict[str, Any]:
        prompt = self._build_extraction_prompt(text, "all")
        response = await self._call_llm(prompt)
        return self._parse_llm_response(response, "all")

    async def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        try:
            result = await client.ainvoke(prompt)
            return result.content
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return "{}"

    def _build_extraction_prompt(self, text: str, target: str) -> str:
        schema_guides = {
            "skills": (
                'Extract skills as JSON array: '
                '[{"name": "...", "category": "...", "proficiency": "..."}]'
            ),
            "experience": (
                'Extract work experience as JSON array: '
                '[{"title": "...", "company": "...", "start_date": "...", '
                '"end_date": "...", "description": "..."}]'
            ),
            "education": (
                'Extract education as JSON array: '
                '[{"institution": "...", "degree": "...", "field": "...", '
                '"start_year": ..., "end_year": ...}]'
            ),
            "all": (
                'Extract all fields: '
                '{"skills": [...], "experience": [...], "education": [...]}'
            ),
        }
        guide = schema_guides.get(target, schema_guides["all"])
        prefix = "Extract structured data from this resume text. "
        suffix = f"\n\n{guide}\n\nResume:\n{text[:4000]}"
        return prefix + "Return ONLY valid JSON, no other text." + suffix

    def _parse_llm_response(self, response: str, target: str) -> Any:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response for {target}")
            return {} if target == "all" else []
