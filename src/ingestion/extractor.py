from __future__ import annotations

import json
import logging
import re
from typing import Any, cast

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
        return cast("list[dict[str, Any]]", self._parse_llm_response(response, "skills"))

    async def extract_experience(self, text: str) -> list[dict[str, Any]]:
        prompt = self._build_extraction_prompt(text, "experience")
        response = await self._call_llm(prompt)
        return cast("list[dict[str, Any]]", self._parse_llm_response(response, "experience"))

    async def extract_education(self, text: str) -> list[dict[str, Any]]:
        prompt = self._build_extraction_prompt(text, "education")
        response = await self._call_llm(prompt)
        return cast("list[dict[str, Any]]", self._parse_llm_response(response, "education"))

    async def extract_all(self, text: str) -> dict[str, Any]:
        prompt = self._build_extraction_prompt(text, "all")
        response = await self._call_llm(prompt)
        return cast("dict[str, Any]", self._parse_llm_response(response, "all"))

    async def _call_llm(self, prompt: str) -> str:
        client = self._get_client()
        try:
            result = await client.ainvoke(prompt)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            # Empty string, not "{}": "{}" parsed to a dict for the list targets.
            return ""
        return _content_to_text(getattr(result, "content", result))

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
        """Parse an LLM reply into the shape the target promises.

        list targets -> list of dicts (deduplicated); "all" -> dict with list-valued
        "skills"/"experience"/"education". Never raises on malformed output.
        """
        data = _load_json(response)
        if data is _MISSING:
            logger.warning(f"Failed to parse LLM response for {target}")
        if target in _SECTIONS:
            return _normalise_section(data, target)
        out: dict[str, Any] = dict(data) if isinstance(data, dict) else {}
        for section in _SECTIONS:
            out[section] = _normalise_section(out.get(section), section, unwrap=False)
        return out


_SECTIONS = ("skills", "experience", "education")
# Fields that identify a bare object as a single record of that section.
_RECORD_KEYS = {
    "skills": ("name",),
    "experience": ("title", "company"),
    "education": ("institution", "degree"),
}
_MISSING: Any = object()
_FENCE_RE = re.compile(r"```[ \t]*(?:json)?[ \t]*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def _content_to_text(content: Any) -> str:
    """LangChain message content may be a str, None, or a list of content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def _load_json(response: str) -> Any:
    cleaned = (response or "").lstrip("\ufeff").strip()
    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    if not cleaned:
        return _MISSING
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Chatty reply ("Sure! [...]"): decode from the first JSON container onwards.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch in "[{":
            try:
                return decoder.raw_decode(cleaned, i)[0]
            except json.JSONDecodeError:
                continue
    return _MISSING


def _normalise_section(data: Any, section: str, unwrap: bool = True) -> list[dict[str, Any]]:
    if unwrap and isinstance(data, dict):
        if isinstance(data.get(section), list):
            data = data[section]
        elif any(k in data for k in _RECORD_KEYS[section]):
            data = [data]
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        if section == "skills":
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            key = " ".join(name.split()).casefold()
        else:
            key = json.dumps(item, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
