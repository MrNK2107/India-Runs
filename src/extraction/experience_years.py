from __future__ import annotations

import re
from typing import Any

from src.extraction.career_history_utils import compute_years_from_dates

_YEARS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(\d+)\+?\s*(?:yrs?|years?|yoe|years?\s+of\s+experience)", re.IGNORECASE),
    re.compile(r"(\d+)\+?\s*(?:years?\s+exp)", re.IGNORECASE),
    re.compile(r"experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)", re.IGNORECASE),
    re.compile(r"(\d+)\s*\+\s*years?", re.IGNORECASE),
    re.compile(r"~?\s*(\d+)\s*(?:years?|yrs?)", re.IGNORECASE),
]

_AGREEMENT_RATIO = 0.2


def extract_experience_years(
    prof: dict[str, Any],
    history: list[dict[str, Any]],
) -> tuple[float | None, str]:
    direct = _safe_float(prof.get("years_of_experience"))

    dates_result, num_valid, dates_conf = compute_years_from_dates(history)

    source_a = direct
    source_b = dates_result

    if source_a is not None and source_b is not None:
        if abs(source_a - source_b) / max(source_a, source_b, 0.1) <= _AGREEMENT_RATIO:
            return round((source_a + source_b) / 2, 1), "average"
        if dates_conf == "high":
            return source_b, "dates_structured"
        return source_a, "direct_structured"

    if source_a is not None:
        return source_a, "direct_structured"

    if source_b is not None:
        return source_b, "dates_structured"

    regex_val = _extract_from_text(prof.get("headline", ""), prof.get("summary", ""))
    if regex_val is not None:
        return regex_val, "regex_fallback"

    return None, "not_found"


def _extract_from_text(*texts: str) -> float | None:
    for text in texts:
        if not text:
            continue
        for pattern in _YEARS_PATTERNS:
            m = pattern.search(text)
            if m:
                val = float(m.group(1))
                if val >= 100:
                    continue
                return val
    return None


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        if f > 100:
            return None
        return f
    except (ValueError, TypeError):
        return None
