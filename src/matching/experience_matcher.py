from __future__ import annotations


class ExperienceMatcher:
    def match(
        self,
        required_min_years: float | None = None,
        required_max_years: float | None = None,
        candidate_years: float | None = None,
        required_industry: str | None = None,
        candidate_industry: str | None = None,
    ) -> float:
        years_score = self._score_years(required_min_years, required_max_years, candidate_years)
        industry_score = self._score_industry(required_industry, candidate_industry)

        if years_score is None and industry_score is None:
            return 1.0
        if years_score is None:
            return industry_score if industry_score is not None else 1.0
        if industry_score is None:
            return years_score

        return 0.7 * years_score + 0.3 * industry_score

    def _score_years(
        self,
        required_min: float | None = None,
        required_max: float | None = None,
        candidate: float | None = None,
    ) -> float | None:
        if required_min is None and required_max is None:
            return None
        if candidate is None:
            return 0.5

        if required_min is not None and candidate < required_min:
            deficit = required_min - candidate
            return max(0.0, 1.0 - deficit / max(required_min, 1))

        if required_max is not None and candidate > required_max:
            excess = candidate - required_max
            return max(0.0, 1.0 - excess / max(required_max, 1))

        return 1.0

    def _score_industry(
        self, required: str | None = None, candidate: str | None = None,
    ) -> float | None:
        if required is None:
            return None
        if candidate is None:
            return 0.5
        return 1.0 if required.lower() == candidate.lower() else 0.3
