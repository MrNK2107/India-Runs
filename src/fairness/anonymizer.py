from __future__ import annotations

import hashlib
import re
from typing import Any

from src.core.constants import IIT_NIT_BITS, METRO_CITIES
from src.core.models import Profile

# Overused LLM-style verbs to strip during style anonymization
LLM_OVERUSED_VERBS: set[str] = {
    "spearheaded", "fostered", "architected", "orchestrated", "pioneered",
    "championed", "drove", "delivered", "revolutionized", "transformed",
    "optimized", "streamlined", "facilitated", "catalyzed", "accelerated",
    "amplified", "supercharged", "evangelized", "synergized", "leveraged",
    "utilized", "implemented", "established", "spearheading", "spearheads",
}

LLM_POWER_PHRASES: list[re.Pattern[str]] = [
    re.compile(r"\bpassionate about\b", re.IGNORECASE),
    re.compile(r"\bproven track record\b", re.IGNORECASE),
    re.compile(r"\bresults[-\s]oriented\b", re.IGNORECASE),
    re.compile(r"\bteam player\b", re.IGNORECASE),
    re.compile(r"\bthink outside the box\b", re.IGNORECASE),
    re.compile(r"\bgame[-\s]changer\b", re.IGNORECASE),
    re.compile(r"\bthought[-\s]leader\b", re.IGNORECASE),
    re.compile(r"\bgoing above and beyond\b", re.IGNORECASE),
    re.compile(r"\bdeep[-\s]dive\b", re.IGNORECASE),
    re.compile(r"\bbest in class\b", re.IGNORECASE),
    re.compile(r"\bmission[-\s]critical\b", re.IGNORECASE),
]


def anonymize_profile(profile: Profile) -> dict[str, Any]:
    """Strip PII from a profile, preserving skills, experience years, industries."""
    name_hash = hashlib.sha256(
        (profile.personal.name + profile.profile_id).encode()
    ).hexdigest()[:8]

    result = {
        "anonymized_id": f"Candidate-{name_hash}",
        "has_degree": len(profile.education) > 0,
        "education_tier": _get_education_tier(profile),
        "experience_years": (
            profile.professional.total_experience_years
            if profile.professional
            else None
        ),
        "seniority_level": (
            profile.professional.seniority_level if profile.professional else None
        ),
        "industry": (
            profile.professional.industry if profile.professional else None
        ),
        "skills": [s.name for s in profile.skills],
        "skill_count": len(profile.skills),
        "total_roles": len(profile.experience),
        "employment_type": (
            profile.professional.employment_type if profile.professional else None
        ),
        "is_passive": profile.signals.is_passive if profile.signals else False,
        "has_portfolio": profile.signals.has_portfolio if profile.signals else False,
        "certifications": profile.signals.certifications if profile.signals else [],
        "languages_spoken": profile.personal.languages_spoken if profile.personal else [],
        "location_tier": _get_location_tier(profile),
        "_anonymization_note": (
            "PII stripped before LLM evaluation "
            "— no name, university, or location data was visible to the evaluator."
        ),
    }
    return result


def _get_education_tier(profile: Profile) -> str:
    """Classify education into tier-1, tier-2, or unknown."""
    for edu in profile.education:
        inst = (edu.institution or "").lower()
        if any(prefix in inst for prefix in IIT_NIT_BITS):
            return "tier-1"
    if len(profile.education) > 0:
        return "tier-2"
    return "unknown"


def _get_location_tier(profile: Profile) -> str:
    """Classify location into metro, tier-2, tier-3, or unknown."""
    city = ""
    if profile.personal and profile.personal.location:
        city = (profile.personal.location.city or "").lower()
    if not city:
        for exp in profile.experience:
            if exp.location:
                city = exp.location.lower().split(",")[0].strip()
                break
    if city in METRO_CITIES:
        return "metro"
    if city:
        return "tier-2/3"
    return "unknown"


def style_anonymize(text: str) -> str:
    """Normalize LLM-writing style artifacts from profile text."""
    for verb in LLM_OVERUSED_VERBS:
        text = re.sub(
            rf"\b{re.escape(verb)}\b",
            _replace_llm_verb(verb),
            text,
            flags=re.IGNORECASE,
        )
    for pattern in LLM_POWER_PHRASES:
        text = pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _replace_llm_verb(verb: str) -> str:
    replacements: dict[str, str] = {
        "spearheaded": "led",
        "fostered": "built",
        "architected": "designed",
        "orchestrated": "coordinated",
        "pioneered": "started",
        "championed": "promoted",
        "drove": "led",
        "delivered": "completed",
        "revolutionized": "improved",
        "transformed": "changed",
        "optimized": "improved",
        "streamlined": "simplified",
        "facilitated": "helped",
        "catalyzed": "sparked",
        "accelerated": "sped up",
        "amplified": "increased",
        "supercharged": "boosted",
        "evangelized": "promoted",
        "synergized": "combined",
        "leveraged": "used",
        "utilized": "used",
        "implemented": "built",
        "established": "set up",
    }
    return replacements.get(verb.lower(), verb)


def anonymize_text_for_bias(text: str) -> str:
    """Quick PII redaction for text before LLM evaluation."""
    text = re.sub(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,}\b",
        "[NAME]", text,
    )
    # Match names with initials (e.g., "J. K. Rowling", "J.K. Rowling", "P. S. Kumar")
    text = re.sub(
        r"\b(?:[A-Z]\.?\s*)[A-Z]\.?\s+[A-Z][a-z]+\b",
        "[NAME]", text,
    )
    text = re.sub(r"\b(?:IIT|NIT|IIIT|BITS)\s+\w+\b", "[UNIVERSITY]", text)
    return text
