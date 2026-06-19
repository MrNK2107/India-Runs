from __future__ import annotations

from typing import Any

from src.extraction.career_history_utils import latest_role

_SENIORITY_TITLES: dict[str, int] = {
    "intern": 0,
    "trainee": 0,
    "junior": 1,
    "jr": 1,
    "mid": 2,
    "mid-level": 2,
    "senior": 3,
    "sr": 3,
    "lead": 4,
    "staff": 4,
    "principal": 5,
    "architect": 5,
    "director": 6,
    "head": 6,
    "vp": 6,
    "chief": 6,
    "cto": 6,
}

_YEARS_JUNIOR = 2
_YEARS_MID = 5
_YEARS_SENIOR = 8
_YEARS_LEAD = 12

_DOMAIN_OVERRIDES: dict[str, int] = {
    "professor": 6,
    "assistant professor": 5,
    "associate professor": 6,
    "lecturer": 3,
    "principal": 5,
    "director": 6,
}

DOMAIN_SPECIFIC_CONSTANTS_NOTE = """
Seniority thresholds (<2 junior, <5 mid, <8 senior, <12 lead) are
tuned for the Indian IT market — the primary data source. These
constants should be revisited if the normalizer is used for
academia, government, or other sectors.
"""


def extract_seniority(
    title: str | None,
    years: float | None,
    history: list[dict[str, Any]],
) -> tuple[int | None, str]:
    title_lower = (title or "").lower()

    for keyword, level in _DOMAIN_OVERRIDES.items():
        if keyword in title_lower:
            return level, "domain_override"

    for keyword, level in _SENIORITY_TITLES.items():
        if keyword in title_lower:
            return level, "title_keyword"

    latest = latest_role(history)
    if latest:
        latest_title = (latest.get("title") or "").lower()
        for keyword, level in _SENIORITY_TITLES.items():
            if keyword in latest_title:
                return level, "history_title"

    if years is not None:
        if years < _YEARS_JUNIOR:
            return 1, "years_fallback"
        if years < _YEARS_MID:
            return 2, "years_fallback"
        if years < _YEARS_SENIOR:
            return 3, "years_fallback"
        if years < _YEARS_LEAD:
            return 4, "years_fallback"
        return 5, "years_fallback"

    return None, "not_found"
