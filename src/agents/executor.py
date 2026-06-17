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

    async def execute(
        self,
        parsed: ParsedQuery,
        top_k: int = 50,
        slider_weights: dict[str, float] | None = None,
    ) -> list[MatchResult]:
        search_text = self._query_to_search_text(parsed)

        # RRF-fused hybrid search for ranking order
        hybrid_results = self.hybrid_search.search(search_text, top_k=top_k * 2)

        # Separate vector + BM25 searches for actual similarity scores
        query_vec = self.hybrid_search.embedder.embed_query(search_text)
        vector_raw = self.hybrid_search.vector_search.search(query_vec, top_k=top_k * 2)
        bm25_raw = self.hybrid_search.bm25_search.search(search_text, top_k=top_k * 2)

        # Build score lookup: profile_id → (vec_score, bm25_score)
        vec_scores: dict[str, float] = {
            pid: self._norm_vec_score(s) for pid, s in vector_raw
        }
        bm25_scores: dict[str, float] = {
            pid: self._norm_bm25_score(s, bm25_raw) for pid, s in bm25_raw
        }

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

            skills_set = set(s.name.lower() for s in profile.skills)
            req_set = set(rs.name.lower() for rs in parsed.required_skills)
            pref_set = set(ps.name.lower() for ps in parsed.preferred_skills)
            all_req = req_set | pref_set
            skill_overlap = len(all_req & skills_set) / max(len(all_req), 1)

            total_years = (
                profile.professional.total_experience_years
                if profile.professional and profile.professional.total_experience_years
                else 0
            )
            exp_match = min(1.0, total_years / 10.0)

            scores_dict: dict[str, float | None] = {
                "semantic_similarity": vec_scores.get(pid),
                "keyword_match": bm25_scores.get(pid),
                "skill_match": skill_overlap,
                "experience_match": exp_match,
                "location_match": None,
                "education_match": None,
                "cross_encoder_score": rerank_score,
            }

            match_scores = self.scorer.compute_overall(scores_dict, slider_weights)

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
            profile = self.profiles.get(pid)
            if profile is None:
                filtered.append((pid, score))
            elif filter_obj.passes(profile):
                filtered.append((pid, score))

        return filtered
