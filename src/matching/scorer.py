from __future__ import annotations

import numpy as np

from src.core.config import get_scoring_config
from src.core.models import MatchScores

DIM_TO_ACTUAL: dict[str, str] = {
    "skill_match": "skill_match",
    "experience_match": "experience_match",
    "education_match": "education_match",
    "assessment_score": "cross_encoder_score",
    "behavioral_signals": "behavioral_signals",
    "cultural_fit": "cultural_fit",
}


DEFAULT_SLIDER_WEIGHTS: dict[str, float] = {
    "skill_match": 0.30,
    "experience_match": 0.25,
    "education_match": 0.15,
    "assessment_score": 0.15,
    "behavioral_signals": 0.10,
    "cultural_fit": 0.05,
}


class CandidateScorer:
    def __init__(self) -> None:
        config = get_scoring_config()
        self.weights = config["scoring_weights"]
        self.slider_defaults = config.get("slider_weights", DEFAULT_SLIDER_WEIGHTS)

    def compute_overall(
        self,
        scores: dict[str, float | None],
        slider_weights: dict[str, float] | None = None,
    ) -> MatchScores:
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
            "behavioral_signals": scores.get("behavioral_signals"),
            "cultural_fit": scores.get("cultural_fit"),
        }

        effective_weights: dict[str, float] = {}
        if slider_weights:
            for slider_dim, actual_dim in DIM_TO_ACTUAL.items():
                if slider_dim in slider_weights and slider_weights[slider_dim] > 0:
                    effective_weights[actual_dim] = slider_weights[slider_dim] / 100.0
            for dim, w in self.slider_defaults.items():
                actual = DIM_TO_ACTUAL.get(dim, dim)
                if actual not in effective_weights:
                    effective_weights[actual] = w
        else:
            effective_weights = dict(self.weights)

        for dim, score in dims.items():
            if score is not None:
                weight = effective_weights.get(dim, 0.0)
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
