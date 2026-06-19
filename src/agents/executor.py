from __future__ import annotations

import logging

from src.core.models import (
    MatchMetadata,
    MatchResult,
    ParsedQuery,
    SearchFilters,
    SearchMethod,
    Skill,
)
from src.core.profile_store import ProfileStore
from src.matching.scorer import CandidateScorer
from src.matching.skill_matcher import SKILL_ALIASES, SkillMatcher
from src.matching.behavioral_scorer import (
    compute_behavioral_score,
    compute_career_trajectory,
    compute_skill_proficiency,
    detect_honeypot,
)
from src.search.filters import SearchFilter
from src.search.hybrid import HybridSearch
from src.search.reranker import CrossEncoderReranker

def _skill_match_score(
    required_names: list[str], profile_skills: list[Skill], raw_text: str | None = None,
) -> float:
    if not required_names:
        return 1.0
    _matcher = SkillMatcher(similarity_threshold=0.8)
    matched = 0
    for rn in required_names:
        result = _matcher.find_best_match(rn, profile_skills)
        if result is not None:
            matched += 1
            continue
        raw_lower = raw_text.lower() if raw_text else ""
        rn_lower = rn.lower()
        if rn_lower in raw_lower:
            matched += 1
            continue
        aliases = SKILL_ALIASES.get(rn_lower, [])
        if any(a in raw_lower for a in aliases):
            matched += 1
    return matched / len(required_names)


def _match_skills_detail(
    required_names: list[str], profile_skills: list[Skill], raw_text: str | None = None,
) -> tuple[list[str], list[str]]:
    matched: list[str] = []
    missing: list[str] = []
    _matcher = SkillMatcher(similarity_threshold=0.8)
    for rn in required_names:
        result = _matcher.find_best_match(rn, profile_skills)
        if result is not None:
            matched.append(rn)
            continue
        raw_lower = raw_text.lower() if raw_text else ""
        rn_lower = rn.lower()
        if rn_lower in raw_lower:
            matched.append(rn)
            continue
        aliases = SKILL_ALIASES.get(rn_lower, [])
        if any(a in raw_lower for a in aliases):
            matched.append(rn)
            continue
        missing.append(rn)
    return matched, missing

logger = logging.getLogger(__name__)


class ExecutorAgent:
    def __init__(
        self,
        hybrid_search: HybridSearch,
        reranker: CrossEncoderReranker,
        scorer: CandidateScorer,
        profiles: ProfileStore,
    ) -> None:
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.scorer = scorer
        self.profile_store = profiles
        self._rerank_top_k = 100

    async def execute(
        self,
        parsed: ParsedQuery,
        top_k: int = 50,
        slider_weights: dict[str, float] | None = None,
    ) -> list[MatchResult]:
        search_text = self._query_to_search_text(parsed)

        hybrid_results = self.hybrid_search.search(search_text, top_k=top_k * 2)

        query_vec = self.hybrid_search.embedder.embed_query(search_text)
        vector_raw = self.hybrid_search.vector_search.search(query_vec, top_k=top_k * 2)
        bm25_raw = self.hybrid_search.bm25_search.search(search_text, top_k=top_k * 2)

        vec_scores: dict[str, float] = {
            pid: self._norm_vec_score(s) for pid, s in vector_raw
        }
        bm25_scores: dict[str, float] = {
            pid: self._norm_bm25_score(s, bm25_raw) for pid, s in bm25_raw
        }

        filtered = self._apply_filters(hybrid_results, parsed)

        rerank_candidates: list[tuple[str, str, float]] = []
        for pid, score in filtered[: self._rerank_top_k]:
            profile = self.profile_store.get(pid)
            if profile is not None:
                rerank_candidates.append((pid, profile.raw_text[:2000], score))
            else:
                rerank_candidates.append((pid, "", score))

        reranked = self.reranker.rerank(search_text, rerank_candidates, top_k=top_k)

        results: list[MatchResult] = []
        for _rank, (pid, rerank_score) in enumerate(reranked, start=1):
            profile = self.profile_store.get(pid)
            if profile is None:
                continue

            req_names = [rs.name for rs in parsed.required_skills]
            pref_names = [ps.name for ps in parsed.preferred_skills]
            all_req = req_names + pref_names
            skill_overlap = _skill_match_score(all_req, profile.skills, profile.raw_text)

            total_years = (
                profile.professional.total_experience_years
                if profile.professional and profile.professional.total_experience_years
                else 0
            )
            exp_match = min(1.0, total_years / 10.0)

            # Honeypot penalty — impossible profiles get heavy penalty but stay in set
            honeypot_reason = detect_honeypot(profile)
            honeypot_penalty = 0.15 if honeypot_reason else 1.0

            # Behavioral & career signals
            behavioral_score = compute_behavioral_score(profile.signals)
            career_trajectory = compute_career_trajectory(profile)
            skill_prof = compute_skill_proficiency(profile, all_req)

            scores_dict: dict[str, float | None] = {
                "semantic_similarity": vec_scores.get(pid),
                "keyword_match": bm25_scores.get(pid),
                "skill_match": skill_overlap * honeypot_penalty,
                "experience_match": exp_match * honeypot_penalty,
                "location_match": None,
                "education_match": None,
                "cross_encoder_score": rerank_score * honeypot_penalty if rerank_score else None,
                "behavioral_score": behavioral_score * honeypot_penalty,
                "career_trajectory_score": career_trajectory * honeypot_penalty,
                "skill_proficiency_score": skill_prof * honeypot_penalty,
            }

            match_scores = self.scorer.compute_overall(scores_dict, slider_weights)

            matched_skills, missing_skills = _match_skills_detail(
                req_names, profile.skills, profile.raw_text,
            )

            loc = profile.personal.location
            city = loc.city if profile.personal and loc else None

            results.append(
                MatchResult(
                    query_id="",
                    profile_id=pid,
                    rank=0,  # will reassign after sorting
                    name=profile.personal.name if profile.personal else "",
                    current_title=(
                        profile.professional.current_title if profile.professional else None
                    ),
                    current_company=(
                        profile.professional.current_company if profile.professional else None
                    ),
                    location=city,
                    experience_years=(
                        profile.professional.total_experience_years
                        if profile.professional else None
                    ),
                    scores=match_scores,
                    matched_skills=list(set(matched_skills)),
                    missing_skills=list(set(missing_skills)),
                    metadata=MatchMetadata(search_method=SearchMethod.HYBRID, reranked=True),
                )
            )

        # Sort by overall score descending, tie-break by profile_id
        results.sort(key=lambda r: (-r.scores.overall, r.profile_id))
        for rank, r in enumerate(results, start=1):
            r.rank = rank

        return results

    @staticmethod
    def _norm_vec_score(score: float) -> float:
        return max(0.0, min(1.0, (score + 1.0) / 2.0))

    @staticmethod
    def _norm_bm25_score(score: float, all_results: list[tuple[str, float]]) -> float:
        if not all_results:
            return 0.0
        max_score = max(s for _, s in all_results)
        if max_score <= 0:
            return 0.0
        return max(0.0, min(1.0, score / max_score))

    def _query_to_search_text(self, parsed: ParsedQuery) -> str:
        parts: list[str] = []
        for rs in parsed.required_skills:
            parts.append(rs.name)
        for ps in parsed.preferred_skills:
            parts.append(ps.name)
        if parsed.experience.min_years:
            parts.append(f"{int(parsed.experience.min_years)}+ years experience")
        if parsed.location.city:
            parts.append(parsed.location.city)
        if parsed.location.remote_ok:
            parts.append("remote")
        return " ".join(parts) if parts else "software engineer"

    def _apply_filters(
        self, results: list[tuple[str, float]], parsed: ParsedQuery,
    ) -> list[tuple[str, float]]:
        filters = SearchFilters(
            location=parsed.location.city,
            min_experience_years=parsed.experience.min_years,
            max_experience_years=parsed.experience.max_years,
            remote_ok=parsed.location.remote_ok,
            exclude_companies=parsed.filters.exclude_companies,
            include_companies=parsed.filters.include_companies,
        )
        filter_obj = SearchFilter(filters)

        filtered: list[tuple[str, float]] = []
        for pid, score in results:
            profile = self.profile_store.get(pid)
            if profile is None:
                filtered.append((pid, score))
            elif filter_obj.passes(profile):
                filtered.append((pid, score))

        return filtered
