from __future__ import annotations

import re

from src.core.models import Profile


def has_encoding_artifacts(text: str) -> bool:
    mojibake_patterns = [
        r"Ã[-¿]",
        r"â[-¿]{2}",
        r"æ[-¿]",
        r"ƒ[-¿]",
        r"\ufffd",
    ]
    return any(re.search(p, text) for p in mojibake_patterns)


def compute_data_quality_score(profile: Profile) -> float:
    score = 0.0

    if profile.personal.name:
        score += 0.10
    if profile.professional.current_title:
        score += 0.10
    if profile.skills:
        score += 0.15
    if profile.experience:
        score += 0.15
    if profile.education:
        score += 0.10
    loc = profile.personal.location
    if loc.city or loc.state:
        score += 0.10

    raw = profile.raw_text
    if len(raw) > 200:
        score += 0.10
    if len(raw) > 500:
        score += 0.05
    if not has_encoding_artifacts(raw):
        score += 0.05
    if any(s.evidence for s in profile.skills):
        score += 0.10

    return min(score, 1.0)


def bulk_score(profiles: list[Profile]) -> list[float]:
    scores = []
    for p in profiles:
        score = compute_data_quality_score(p)
        p.metadata.data_quality_score = score
        scores.append(score)
    return scores
