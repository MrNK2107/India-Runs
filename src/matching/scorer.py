from __future__ import annotations

import numpy as np

from src.core.config import get_scoring_config
from src.core.models import MatchScores


class CandidateScorer:
    def __init__(self) -> None:
        config = get_scoring_config()
        self.weights = config["scoring_weights"]

    def compute_overall(self, scores: dict[str, float | None]) -> MatchScores:
        total_weight = 0.0
        weighted_sum = 0.0
        components: dict[str, float | None] = {}

        dims = {
            "semantic_similarity": scores.get("semantic_similarity"),
            "keyword_match": scores.get("keyword_match"),
            "skill_match": scores.get("skill_match"),
            "experience_match": scores.get("experience_match"),
            "location_match": scores.get("location_match"),
            "education_match": scores.get("education_match"),
            "cross_encoder_score": scores.get("cross_encoder_score"),
        }

        for dim, score in dims.items():
            if score is not None:
                weight = self.weights.get(dim, 0.05)
                total_weight += weight
                weighted_sum += weight * score
            components[dim] = score

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        overall = max(0.0, min(1.0, overall))

        confidence = self.compute_confidence({k: v for k, v in components.items() if v is not None})

        return MatchScores(
            overall=overall,
            semantic_similarity=components.get("semantic_similarity") or 0.0,
            keyword_match=components.get("keyword_match") or 0.0,
            skill_match=components.get("skill_match") or 0.0,
            experience_match=components.get("experience_match") or 0.0,
            location_match=components.get("location_match"),
            education_match=components.get("education_match"),
            cross_encoder_score=components.get("cross_encoder_score"),
            confidence=confidence,
        )

    def compute_confidence(self, scores: dict[str, float | None]) -> float:
        values = [v for v in scores.values() if v is not None]
        if len(values) < 2:
            return 0.5
        std = float(np.std(values))
        return max(0.0, min(1.0, 1.0 - std))
