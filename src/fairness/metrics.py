from __future__ import annotations

from typing import Any

import numpy as np

from src.core.models import MatchResult, Profile

TIER_1_CITIES = {
    "bangalore", "bengaluru", "hyderabad", "pune", "chennai",
    "mumbai", "kolkata", "noida", "gurgaon", "gurugram",
    "delhi", "new delhi",
}

IIT_NIT_BITS = {"iit", "nit", "bits", "iiit"}


def _get_city_tier(profile: Profile) -> str | None:
    city = ""
    if profile.personal and profile.personal.location:
        city = (profile.personal.location.city or "").lower()
    if not city:
        for exp in profile.experience:
            if exp.location:
                city = exp.location.lower()
                break
    if not city:
        return None
    return "tier1" if city in TIER_1_CITIES else "tier2"


def _is_top_university(profile: Profile) -> bool:
    for edu in profile.education:
        inst = (edu.institution or "").lower()
        if any(prefix in inst for prefix in IIT_NIT_BITS):
            return True
    return False


def compute_demographic_parity(
    matches: list[MatchResult], profiles: dict[str, Profile], protected_attribute: str,
) -> float:
    if not matches:
        return 1.0

    top_100 = matches[:100]
    total = len(top_100)
    if total == 0:
        return 1.0

    if protected_attribute == "university":
        protected_count = sum(1 for m in top_100 if _is_top_university(profiles.get(m.profile_id)))
        unprotected_count = total - protected_count
    elif protected_attribute in ("city", "location"):
        protected_count = sum(
            1 for m in top_100
            if _get_city_tier(profiles.get(m.profile_id)) == "tier2"
        )
        unprotected_count = total - protected_count
    elif protected_attribute == "language":
        protected_count = sum(
            1 for m in top_100
            if (profiles.get(m.profile_id) or Profile).metadata.language_detected != "en"
        )
        unprotected_count = total - protected_count
    else:
        return 1.0

    if unprotected_count == 0:
        return 1.0

    protected_rate = protected_count / total
    unprotected_rate = unprotected_count / total
    if unprotected_rate == 0:
        return 1.0
    return min(1.0, protected_rate / unprotected_rate)


def compute_disparate_impact_ratio(
    matches: list[MatchResult],
    profiles: dict[str, Profile],
    protected_group: str,
    majority_group: str,
) -> float:
    top_100 = matches[:100]
    if not top_100:
        return 1.0

    def _in_group(m: MatchResult, group: str) -> bool:
        profile = profiles.get(m.profile_id)
        if profile is None:
            return False
        group_lower = group.lower()
        for edu in profile.education:
            inst = (edu.institution or "").lower()
            if group_lower == "top_university" and any(p in inst for p in IIT_NIT_BITS):
                return True
            if group_lower == "other_university" and not any(p in inst for p in IIT_NIT_BITS):
                return True
            if group_lower in inst:
                return True
        return False

    protected_selected = sum(1 for m in top_100 if _in_group(m, protected_group))
    majority_selected = sum(1 for m in top_100 if _in_group(m, majority_group))

    if majority_selected == 0:
        return 1.0

    ratio = protected_selected / majority_selected if majority_selected > 0 else 1.0
    return min(1.0, ratio)


def compute_language_bias(
    matches: list[MatchResult], profiles: dict[str, Profile],
) -> dict[str, float]:
    en_scores: list[float] = []
    non_en_scores: list[float] = []

    for m in matches:
        profile = profiles.get(m.profile_id)
        if profile is None:
            continue
        lang = profile.metadata.language_detected if profile.metadata else "en"
        if lang == "en":
            en_scores.append(float(m.rank))
        else:
            non_en_scores.append(float(m.rank))

    en_avg = float(np.mean(en_scores)) if en_scores else 0.0
    non_en_avg = float(np.mean(non_en_scores)) if non_en_scores else 0.0

    return {
        "en_avg_rank": en_avg,
        "non_en_avg_rank": non_en_avg,
        "rank_diff": non_en_avg - en_avg,
        "en_count": len(en_scores),
        "non_en_count": len(non_en_scores),
    }


def compute_location_bias(
    matches: list[MatchResult], profiles: dict[str, Profile],
) -> dict[str, float]:
    tier1_ranks: list[float] = []
    tier2_ranks: list[float] = []

    for m in matches:
        profile = profiles.get(m.profile_id)
        if profile is None:
            continue
        tier = _get_city_tier(profile)
        if tier == "tier1":
            tier1_ranks.append(float(m.rank))
        elif tier == "tier2":
            tier2_ranks.append(float(m.rank))

    t1_avg = float(np.mean(tier1_ranks)) if tier1_ranks else 0.0
    t2_avg = float(np.mean(tier2_ranks)) if tier2_ranks else 0.0

    return {
        "tier1_avg_rank": t1_avg,
        "tier2_avg_rank": t2_avg,
        "rank_diff": t2_avg - t1_avg,
        "tier1_count": len(tier1_ranks),
        "tier2_count": len(tier2_ranks),
    }


def compute_all_fairness_metrics(
    matches: list[MatchResult], profiles: dict[str, Profile],
) -> dict[str, Any]:
    return {
        "demographic_parity": {
            "university": compute_demographic_parity(matches, profiles, "university"),
            "city": compute_demographic_parity(matches, profiles, "city"),
            "language": compute_demographic_parity(matches, profiles, "language"),
        },
        "language_bias": compute_language_bias(matches, profiles),
        "location_bias": compute_location_bias(matches, profiles),
        "total_candidates_analyzed": len(matches),
    }
