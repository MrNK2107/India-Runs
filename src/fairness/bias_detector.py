from __future__ import annotations

from typing import Any

from src.core.models import MatchResult, Profile

TIER_1_CITIES = {
    "bangalore", "bengaluru", "hyderabad", "pune", "chennai",
    "mumbai", "kolkata", "noida", "gurgaon", "gurugram",
    "delhi", "new delhi",
}

IIT_NIT_BITS = {"iit", "nit", "bits", "iiit"}


class BiasDetector:
    def check_name_bias(
        self, matches: list[MatchResult], profiles: dict[str, Profile],
    ) -> dict[str, Any]:
        if not matches:
            return {"detected": False, "observations": [], "details": {}}

        buckets: dict[str, list[float]] = {}
        for m in matches:
            profile = profiles.get(m.profile_id)
            if profile is None:
                continue
            name = profile.personal.name.strip() if profile.personal else ""
            if not name:
                continue
            prefix = name[0].upper()
            if prefix not in buckets:
                buckets[prefix] = []
            buckets[prefix].append(m.scores.overall)

        observations = []
        for prefix, scores in sorted(buckets.items()):
            if len(scores) >= 3:
                avg = sum(scores) / len(scores)
                observations.append(f"{prefix}: avg score {avg:.3f} (n={len(scores)})")

        return {
            "detected": False,
            "observations": observations,
            "details": {k: len(v) for k, v in buckets.items()},
        }

    def check_language_bias(
        self, matches: list[MatchResult], profiles: dict[str, Profile],
    ) -> dict[str, Any]:
        if not matches:
            return {"detected": False, "observations": [], "details": {}}

        en_scores: list[float] = []
        non_en_scores: list[float] = []
        en_names: list[str] = []
        non_en_names: list[str] = []

        for m in matches:
            profile = profiles.get(m.profile_id)
            if profile is None:
                continue
            lang = (profile.metadata.language_detected or "en") if profile.metadata else "en"
            name = profile.personal.name if profile.personal else m.profile_id
            if lang == "en":
                en_scores.append(m.scores.overall)
                en_names.append(name)
            else:
                non_en_scores.append(m.scores.overall)
                non_en_names.append(name)

        en_avg = sum(en_scores) / len(en_scores) if en_scores else 0
        non_en_avg = sum(non_en_scores) / len(non_en_scores) if non_en_scores else 0
        diff = en_avg - non_en_avg

        return {
            "detected": diff > 0.1,
            "observations": [
                f"English profiles avg score: {en_avg:.3f} (n={len(en_scores)})",
                f"Non-English profiles avg score: {non_en_avg:.3f} (n={len(non_en_scores)})",
                f"Difference: {diff:.3f} {'(potential bias)' if diff > 0.1 else '(acceptable)'}",
            ],
            "details": {"en_avg_score": en_avg, "non_en_avg_score": non_en_avg, "diff": diff},
        }

    def check_location_bias(
        self, matches: list[MatchResult], profiles: dict[str, Profile],
    ) -> dict[str, Any]:
        if not matches:
            return {"detected": False, "observations": [], "details": {}}

        tier1_scores: list[float] = []
        tier2_scores: list[float] = []

        for m in matches:
            profile = profiles.get(m.profile_id)
            if profile is None:
                continue
            city = ""
            if profile.personal and profile.personal.location:
                city = (profile.personal.location.city or "").lower()
            if not city:
                for exp in profile.experience:
                    if exp.location:
                        city = exp.location.lower()
                        break
            if city in TIER_1_CITIES:
                tier1_scores.append(m.scores.overall)
            elif city:
                tier2_scores.append(m.scores.overall)

        t1_avg = sum(tier1_scores) / len(tier1_scores) if tier1_scores else 0
        t2_avg = sum(tier2_scores) / len(tier2_scores) if tier2_scores else 0
        diff = t1_avg - t2_avg

        return {
            "detected": diff > 0.1,
            "observations": [
                f"Tier-1 city profiles avg score: {t1_avg:.3f} (n={len(tier1_scores)})",
                f"Tier-2/3 city profiles avg score: {t2_avg:.3f} (n={len(tier2_scores)})",
                f"Difference: {diff:.3f} {'(potential bias)' if diff > 0.1 else '(acceptable)'}",
            ],
            "details": {"tier1_avg": t1_avg, "tier2_avg": t2_avg, "diff": diff},
        }

    def check_university_bias(
        self, matches: list[MatchResult], profiles: dict[str, Profile],
    ) -> dict[str, Any]:
        if not matches:
            return {"detected": False, "observations": [], "details": {}}

        top_univ_scores: list[float] = []
        other_scores: list[float] = []

        for m in matches:
            profile = profiles.get(m.profile_id)
            if profile is None:
                continue
            is_top_univ = False
            for edu in profile.education:
                inst = (edu.institution or "").lower()
                if any(prefix in inst for prefix in IIT_NIT_BITS):
                    is_top_univ = True
                    break
            if is_top_univ:
                top_univ_scores.append(m.scores.overall)
            else:
                other_scores.append(m.scores.overall)

        top_avg = sum(top_univ_scores) / len(top_univ_scores) if top_univ_scores else 0
        other_avg = sum(other_scores) / len(other_scores) if other_scores else 0
        diff = top_avg - other_avg

        return {
            "detected": diff > 0.1,
            "observations": [
                f"IIT/NIT/BITS profiles avg score: {top_avg:.3f} (n={len(top_univ_scores)})",
                f"Other university profiles avg score: {other_avg:.3f} (n={len(other_scores)})",
                f"Difference: {diff:.3f} {'(potential bias)' if diff > 0.1 else '(acceptable)'}",
            ],
            "details": {"top_univ_avg": top_avg, "other_avg": other_avg, "diff": diff},
        }
