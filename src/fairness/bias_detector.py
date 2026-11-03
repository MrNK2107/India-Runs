from __future__ import annotations

from typing import Any

from src.core.constants import TIER_1_CITIES
from src.core.models import MatchResult, Profile
from src.fairness.groups import city_key, is_top_institution

BIAS_THRESHOLD = 0.1
MIN_BUCKET_SIZE = 3


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _compare_groups(
    label_a: str, scores_a: list[float], label_b: str, scores_b: list[float],
    keys: tuple[str, str],
) -> dict[str, Any]:
    """Mean-score gap between two groups; group a is the (potentially) favoured one.

    With an empty group there is nothing to compare: previously its average counted as 0,
    so e.g. an all-tier-1 shortlist was reported as location bias.
    """
    avg_a, avg_b = _avg(scores_a), _avg(scores_b)
    observations = [
        f"{label_a} avg score: {avg_a:.3f} (n={len(scores_a)})",
        f"{label_b} avg score: {avg_b:.3f} (n={len(scores_b)})",
    ]
    if not scores_a or not scores_b:
        observations.append("Difference: n/a (insufficient data: one group is empty)")
        return {
            "detected": False,
            "observations": observations,
            "details": {keys[0]: avg_a, keys[1]: avg_b, "diff": 0.0},
        }
    diff = avg_a - avg_b
    detected = diff > BIAS_THRESHOLD
    observations.append(
        f"Difference: {diff:.3f} {'(potential bias)' if detected else '(acceptable)'}"
    )
    return {
        "detected": detected,
        "observations": observations,
        "details": {keys[0]: avg_a, keys[1]: avg_b, "diff": diff},
    }


class BiasDetector:
    def detect_bias(
        self, matches: list[MatchResult], profiles: dict[str, Profile],
        bias_type: str = "name",
    ) -> dict[str, Any]:
        """Detect bias of specified type.

        Args:
            matches: List of match results.
            profiles: Dict of profile_id -> Profile.
            bias_type: One of 'name', 'language', 'location', 'university'.

        Returns:
            dict with 'detected', 'observations', 'details' keys.
        """
        if bias_type == "name":
            return self.check_name_bias(matches, profiles)
        elif bias_type == "language":
            return self.check_language_bias(matches, profiles)
        elif bias_type == "location":
            return self.check_location_bias(matches, profiles)
        elif bias_type == "university":
            return self.check_university_bias(matches, profiles)
        else:
            return {
                "detected": False,
                "observations": [f"Unknown bias type: {bias_type}"],
                "details": {},
            }

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

        # A first-letter bucket (n >= MIN_BUCKET_SIZE) whose mean differs from the overall
        # mean by more than BIAS_THRESHOLD is flagged. (Previously nothing was ever flagged.)
        all_scores = [x for v in buckets.values() for x in v]
        overall = _avg(all_scores)
        observations = []
        for prefix, scores in sorted(buckets.items()):
            if len(scores) >= MIN_BUCKET_SIZE:
                avg = _avg(scores)
                flag = " (potential bias)" if abs(avg - overall) > BIAS_THRESHOLD else ""
                observations.append(f"{prefix}: avg score {avg:.3f} (n={len(scores)}){flag}")

        return {
            "detected": any(obs.endswith("(potential bias)") for obs in observations),
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

        return _compare_groups(
            "English profiles", en_scores, "Non-English profiles", non_en_scores,
            ("en_avg_score", "non_en_avg_score"),
        )

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
                city = city_key(profile.personal.location.city)
            if not city:
                for exp in profile.experience:
                    if exp.location:
                        city = city_key(exp.location)
                        break
            if city in TIER_1_CITIES:
                tier1_scores.append(m.scores.overall)
            elif city:
                tier2_scores.append(m.scores.overall)

        return _compare_groups(
            "Tier-1 city profiles", tier1_scores, "Tier-2/3 city profiles", tier2_scores,
            ("tier1_avg", "tier2_avg"),
        )

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
            is_top_univ = any(is_top_institution(e.institution) for e in profile.education)
            if is_top_univ:
                top_univ_scores.append(m.scores.overall)
            else:
                other_scores.append(m.scores.overall)

        return _compare_groups(
            "IIT/NIT/BITS profiles", top_univ_scores,
            "Other university profiles", other_scores,
            ("top_univ_avg", "other_avg"),
        )
