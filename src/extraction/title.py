from __future__ import annotations

import re
from typing import Any

from src.extraction.career_history_utils import latest_role

_SENIORITY_PREFIXES = [
    "senior", "sr", "junior", "jr", "lead", "staff",
    "principal", "chief", "head", "vp", "vp of", "director of",
    "associate", "assistant", "principal",
]

_SENIORITY_PATTERN = re.compile(
    r"^(?:" + "|".join(_SENIORITY_PREFIXES) + r")\s+",
    re.IGNORECASE,
)


def extract_title(
    prof: dict[str, Any],
    history: list[dict[str, Any]],
) -> tuple[str | None, str]:
    direct = prof.get("current_title")
    if direct and isinstance(direct, str) and direct.strip():
        return direct.strip(), "direct"

    latest = latest_role(history)
    if latest:
        title = latest.get("title")
        if title and isinstance(title, str) and title.strip():
            return title.strip(), "history"

    headline = prof.get("headline", "")
    if headline and isinstance(headline, str) and headline.strip():
        extracted = _parse_headline(headline)
        if extracted:
            return extracted, "headline"

    return None, "not_found"


def _parse_headline(headline: str) -> str | None:
    candidate = headline.strip()
    if "|" in candidate:
        candidate = candidate.split("|")[0].strip()
    if " at " in candidate.lower():
        candidate = candidate.lower().split(" at ")[0].strip().title()
    candidate = _SENIORITY_PATTERN.sub("", candidate).strip()
    if candidate:
        return candidate
    return None
