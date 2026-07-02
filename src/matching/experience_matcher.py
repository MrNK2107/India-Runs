from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.models import Profile


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

    def match_title(self, query: str, profile: Profile) -> float:
        """Score how well the candidate's career titles align with the target query role."""
        query_role = self._extract_role(query)
        if not query_role:
            return 1.0

        current_title = profile.professional.current_title if profile.professional else ""
        current_score = self._score_title_compatibility(query_role, current_title)

        past_scores = []
        for exp in profile.experience:
            past_scores.append(self._score_title_compatibility(query_role, exp.title))

        max_past_score = max(past_scores) if past_scores else 0.0

        # Current title is primary (70%), past experiences provide a fallback (30%).
        return 0.7 * current_score + 0.3 * max_past_score

    def _extract_role(self, query: str) -> str | None:
        query_lower = query.lower()
        # Order by length/specificity to avoid partial matching issues
        roles = [
            "full stack developer", "fullstack developer", "full stack engineer", "fullstack engineer", "full stack", "fullstack",
            "backend developer", "backend engineer", "back end developer", "back end engineer", "backend",
            "frontend developer", "frontend engineer", "front end developer", "front end engineer", "frontend",
            "data scientist", "data science",
            "data engineer", "data engineering",
            "ml engineer", "machine learning engineer", "ml", "machine learning",
            "devops engineer", "devops",
            "solutions architect", "cloud architect", "systems architect", "solutions architect", "architect",
            "java developer", "java engineer",
            "mobile developer", "android developer", "ios developer", "mobile engineer",
            "python developer", "python engineer",
            "engineering manager", "tech lead", "technical lead",
            "cybersecurity engineer", "cybersecurity", "security engineer", "application security", "security analyst",
            "qa automation engineer", "qa engineer", "qa automation", "quality assurance", "qa",
            "data analyst", "business analyst",
            "product manager",
            "site reliability engineer", "sre",
            "software engineer", "software developer", "sde", "developer", "engineer"
        ]
        for r in roles:
            if r in query_lower:
                return r
        return None

    def _get_title_category(self, title: str) -> str:
        title_lower = title.lower()
        if any(w in title_lower for w in ["fullstack", "full stack"]):
            return "fullstack"
        if "frontend" in title_lower or "front end" in title_lower or "ui" in title_lower:
            return "frontend"
        if "backend" in title_lower or "back end" in title_lower:
            return "backend"
        if "devops" in title_lower or "sre" in title_lower or "site reliability" in title_lower or "cloud" in title_lower or "infrastructure" in title_lower:
            return "devops"
        if "data scientist" in title_lower or "data science" in title_lower or "ml" in title_lower or "machine learning" in title_lower or "deep learning" in title_lower or "nlp" in title_lower or "computer vision" in title_lower:
            return "data_ml"
        if "data engineer" in title_lower or "data engineering" in title_lower:
            return "data_eng"
        if "qa" in title_lower or "test" in title_lower or "testing" in title_lower or "quality assurance" in title_lower:
            return "qa"
        if "analyst" in title_lower or "analytics" in title_lower:
            return "analyst"
        if "architect" in title_lower:
            return "architect"
        if "manager" in title_lower or "lead" in title_lower or "head" in title_lower or "director" in title_lower:
            return "management"
        if "java" in title_lower:
            return "backend"
        if "python" in title_lower:
            return "backend"
        if "software engineer" in title_lower or "software developer" in title_lower or "sde" in title_lower or "developer" in title_lower or "engineer" in title_lower:
            return "general_se"
        return "other"

    def _score_title_compatibility(self, query_role: str, candidate_title: str) -> float:
        if not candidate_title:
            return 0.5

        q_cat = self._get_title_category(query_role)
        c_cat = self._get_title_category(candidate_title)

        if q_cat == c_cat:
            return 1.0

        compatibility = {
            "general_se": {
                "fullstack": 1.0, "backend": 1.0, "frontend": 1.0, "general_se": 1.0,
                "architect": 0.9, "management": 0.8,
                "devops": 0.6, "data_eng": 0.6, "data_ml": 0.6,
                "qa": 0.4, "analyst": 0.4, "other": 0.3
            },
            "fullstack": {
                "fullstack": 1.0, "backend": 0.8, "frontend": 0.8, "general_se": 0.9,
                "architect": 0.7, "management": 0.6, "devops": 0.4, "data_eng": 0.4,
                "qa": 0.2, "analyst": 0.2, "other": 0.2
            },
            "backend": {
                "backend": 1.0, "fullstack": 0.9, "general_se": 0.8,
                "architect": 0.7, "devops": 0.5, "data_eng": 0.6,
                "frontend": 0.3, "qa": 0.2, "analyst": 0.2, "other": 0.2
            },
            "frontend": {
                "frontend": 1.0, "fullstack": 0.9, "general_se": 0.8,
                "backend": 0.3, "architect": 0.6, "devops": 0.3,
                "qa": 0.2, "analyst": 0.2, "other": 0.2
            },
            "devops": {
                "devops": 1.0, "general_se": 0.5, "fullstack": 0.4, "backend": 0.4,
                "qa": 0.3, "other": 0.1
            },
            "data_ml": {
                "data_ml": 1.0, "analyst": 0.6, "data_eng": 0.6, "general_se": 0.4, "other": 0.1
            },
            "data_eng": {
                "data_eng": 1.0, "backend": 0.7, "general_se": 0.5, "data_ml": 0.6,
                "analyst": 0.5, "other": 0.1
            },
            "qa": {
                "qa": 1.0, "general_se": 0.4, "other": 0.1
            },
            "analyst": {
                "analyst": 1.0, "data_ml": 0.5, "data_eng": 0.5, "other": 0.1
            },
            "architect": {
                "architect": 1.0, "general_se": 0.8, "management": 0.7, "other": 0.2
            },
            "management": {
                "management": 1.0, "general_se": 0.7, "architect": 0.7, "other": 0.1
            }
        }

        q_rules = compatibility.get(q_cat, {})
        return q_rules.get(c_cat, 0.2)
