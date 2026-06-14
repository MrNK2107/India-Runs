from __future__ import annotations

from src.core.models import MatchRecommendation, Rationale


class RationaleValidator:
    def validate(self, rationale: Rationale) -> tuple[bool, list[str]]:
        issues: list[str] = []

        if len(rationale.summary) < 10:
            issues.append("Summary too short (min 10 chars)")
        if len(rationale.summary) > 500:
            issues.append("Summary too long (max 500 chars)")

        if not rationale.strengths:
            issues.append("No strengths listed")

        valid_recommendations = {r.value for r in MatchRecommendation}
        if rationale.recommendation.value not in valid_recommendations:
            issues.append(f"Invalid recommendation: {rationale.recommendation}")

        return len(issues) == 0, issues

    def validate_batch(self, rationales: list[Rationale]) -> dict[str, int]:
        total = len(rationales)
        valid = 0
        all_issues: list[str] = []
        for r in rationales:
            is_valid, issues = self.validate(r)
            if is_valid:
                valid += 1
            all_issues.extend(issues)
        return {
            "total": total,
            "valid": valid,
            "invalid": total - valid,
            "total_issues": len(all_issues),
        }
