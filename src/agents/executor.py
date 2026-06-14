from __future__ import annotations

import logging

from src.core.models import (
    MatchMetadata,
    MatchResult,
    ParsedQuery,
    Profile,
    SearchFilters,
    SearchMethod,
)
from src.matching.scorer import CandidateScorer
from src.search.filters import SearchFilter
from src.search.hybrid import HybridSearch
from src.search.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class ExecutorAgent:
    def __init__(
        self,
        hybrid_search: HybridSearch,
        reranker: CrossEncoderReranker,
        scorer: CandidateScorer,
        profiles: dict[str, Profile],
    ) -> None:
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.scorer = scorer
        self.profiles = profiles

    async def execute(self, parsed: ParsedQuery, top_k: int = 50) -> list[MatchResult]:
        search_text = self._query_to_search_text(parsed)

        hybrid_results = self.hybrid_search.search(search_text, top_k=top_k * 2)

        filtered = self._apply_filters(hybrid_results, parsed)

        rerank_candidates: list[tuple[str, str, float]] = []
        for pid, score in filtered[:50]:
            profile = self.profiles.get(pid)
            if profile is not None:
                rerank_candidates.append((pid, profile.raw_text[:2000], score))
            else:
                rerank_candidates.append((pid, search_text, score))

        reranked = self.reranker.rerank(search_text, rerank_candidates, top_k=top_k)

        results: list[MatchResult] = []
        for rank, (pid, rerank_score) in enumerate(reranked, start=1):
            profile = self.profiles.get(pid)
            if profile is None:
                continue

            scores_dict: dict[str, float | None] = {
                "semantic_similarity": None,
                "keyword_match": None,
                "skill_match": None,
                "experience_match": None,
                "location_match": None,
                "education_match": None,
                "cross_encoder_score": rerank_score,
            }

            for hpid, hscore in hybrid_results:
                if hpid == pid:
                    scores_dict["semantic_similarity"] = hscore
                    scores_dict["keyword_match"] = hscore
                    break

            match_scores = self.scorer.compute_overall(scores_dict)

            matched_skills = [s.name for s in profile.skills]
            req_names = [rs.name for rs in parsed.required_skills]
            missing_skills = [n for n in req_names if n not in matched_skills]

            loc = profile.personal.location
            city = loc.city if profile.personal and loc else None

            results.append(
                MatchResult(
                    query_id="",
                    profile_id=pid,
                    rank=rank,
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

        return results

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
            profile = self.profiles.get(pid)
            if profile is None:
                filtered.append((pid, score))
            elif filter_obj.passes(profile):
                filtered.append((pid, score))

        return filtered
