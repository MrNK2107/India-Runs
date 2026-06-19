from __future__ import annotations

from difflib import SequenceMatcher

from src.core.models import RequiredSkill, Skill, SkillImportance

SKILL_ALIASES: dict[str, list[str]] = {
    "python": ["python3", "py"],
    "javascript": ["js", "ecmascript", "es6"],
    "typescript": ["ts"],
    "react": ["reactjs", "react.js"],
    "vue": ["vuejs", "vue.js"],
    "angular": ["angularjs", "angular.js", "angular 2+"],
    "node.js": ["nodejs", "node"],
    "kubernetes": ["k8s"],
    "docker": ["docker"],
    "aws": ["amazon web services"],
    "gcp": ["google cloud platform", "google cloud"],
    "azure": ["microsoft azure"],
    "machine learning": ["ml"],
    "artificial intelligence": ["ai"],
    "natural language processing": ["nlp"],
    "ci/cd": ["ci", "cd", "continuous integration", "continuous deployment"],
    "sql": ["mysql", "postgresql", "postgres", "pl/sql"],
    "nosql": ["mongodb", "cassandra", "redis"],
    "git": ["github", "gitlab", "bitbucket"],
    "rest api": ["rest", "restful", "restful api"],
    "html": ["html5"],
    "css": ["css3"],
    "tensorflow": ["tf"],
    "pytorch": ["torch"],
    "fastapi": ["fast api"],
    "django": ["django"],
    "flask": ["flask"],
    "spring boot": ["spring", "spring framework"],
    "data science": ["data science"],
    "deep learning": ["dl"],
    "computer vision": ["cv"],
    "statistics": ["statistical analysis", "statistical modeling"],
    "react native": ["react-native", "reactnative"],
}


class SkillMatcher:
    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self.similarity_threshold = similarity_threshold

    def match_skills(
        self, required: list[RequiredSkill], candidate_skills: list[Skill],
    ) -> tuple[float, list[dict]]:
        if not required:
            return 1.0, []

        total_weight = 0.0
        weighted_score = 0.0
        details: list[dict] = []

        for req in required:
            importance_weight = self._importance_weight(req.importance)
            total_weight += importance_weight

            match = self.find_best_match(req.name, candidate_skills)
            if match is not None:
                prof_score = self.compute_proficiency_match(
                    req.min_proficiency, match.proficiency,
                )
                skill_score = 0.5 + 0.5 * prof_score
                found = True
            else:
                skill_score = 0.0
                found = False

            details.append({
                "skill": req.name,
                "importance": req.importance.value,
                "found": found,
                "score": skill_score,
            })
            weighted_score += importance_weight * skill_score

        overall = weighted_score / total_weight if total_weight > 0 else 0.0
        return overall, details

    def find_best_match(self, required_name: str, candidate_skills: list[Skill]) -> Skill | None:
        normalized_req = self._normalize(required_name)
        aliases = [self._normalize(a) for a in SKILL_ALIASES.get(normalized_req, [])]

        best_score = 0.0
        best_skill: Skill | None = None

        for skill in candidate_skills:
            normalized_skill = self._normalize(skill.name)

            if normalized_req == normalized_skill:
                return skill

            if any(normalized_skill == alias for alias in aliases):
                return skill

            score = self._fuzzy_score(normalized_req, normalized_skill)
            if score > best_score:
                best_score = score
                best_skill = skill

        if best_score >= self.similarity_threshold:
            return best_skill
        return None

    def _normalize(self, name: str) -> str:
        return name.strip().lower()

    def _fuzzy_score(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def compute_proficiency_match(self, required: str | None, candidate: str | None) -> float:
        levels = ["beginner", "intermediate", "advanced", "expert"]
        if required is None or candidate is None:
            return 1.0
        req_idx = levels.index(required.lower()) if required.lower() in levels else 0
        cand_idx = levels.index(candidate.lower()) if candidate.lower() in levels else 0
        if cand_idx >= req_idx:
            return 1.0
        return max(0.0, 1.0 - (req_idx - cand_idx) * 0.25)

    def _importance_weight(self, importance: SkillImportance) -> float:
        weights = {
            SkillImportance.REQUIRED: 1.0,
            SkillImportance.PREFERRED: 0.6,
            SkillImportance.NICE_TO_HAVE: 0.3,
        }
        return weights.get(importance, 0.5)
