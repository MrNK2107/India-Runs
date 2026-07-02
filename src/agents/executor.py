from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from src.core.models import (
    MatchMetadata,
    MatchResult,
    ParsedQuery,
    Profile,
    SearchFilters,
    SearchMethod,
    Skill,
)
from src.core.profile_store import ProfileStore
from src.matching.behavioral_scorer import (
    compute_behavioral_score,
    compute_career_trajectory,
    compute_skill_proficiency,
    detect_honeypot,
)
from src.matching.scorer import CandidateScorer
from src.matching.skill_matcher import SkillMatcher
from src.search.filters import SearchFilter
from src.search.hybrid import HybridSearch
from src.search.reranker import CrossEncoderReranker

# Role-type → representative skills used when the query parser produces no explicit skills.
_ROLE_SUBSKILLS: dict[str, list[str]] = {
    "backend": [
        "node.js", "nodejs", "express", "nestjs", "django", "flask", "fastapi",
        "spring boot", "spring", "golang", "go", "java", "python",
        "postgresql", "mysql", "mongodb", "redis", "graphql", "rest api",
    ],
    "frontend": [
        "react", "reactjs", "vue", "angular", "javascript", "typescript",
        "html", "css", "next.js", "nuxt", "svelte", "tailwind", "webpack", "vite",
    ],
    "fullstack": [
        "react", "vue", "angular", "node.js", "nodejs", "javascript", "typescript",
        "html", "css", "next.js", "sql", "nosql", "postgresql", "mongodb", "graphql",
    ],
    "devops": [
        "ci/cd", "docker", "kubernetes", "k8s", "terraform", "jenkins", "ansible",
        "aws", "prometheus", "grafana", "github actions", "argocd", "helm",
    ],
    "data science": [
        "python", "pandas", "numpy", "scikit-learn", "statistics", "machine learning",
        "sql", "tableau", "tensorflow", "pytorch",
    ],
    "data engineering": [
        "python", "scala", "sql", "spark", "hadoop", "airflow", "kafka", "dbt",
        "snowflake", "redshift", "bigquery", "etl",
    ],
    "ml": [
        "python", "pytorch", "tensorflow", "scikit-learn", "deep learning", "keras",
        "mlops", "nlp", "transformers", "llm",
    ],
    "mobile": [
        "android", "ios", "flutter", "react native", "swift", "kotlin", "dart",
    ],
}

# Ordered longest-first to avoid partial matches (e.g. "backend" before "back")
_ROLE_KEYWORDS: list[tuple[str, str]] = [
    ("full stack", "fullstack"), ("fullstack", "fullstack"),
    ("back end", "backend"), ("backend", "backend"),
    ("front end", "frontend"), ("frontend", "frontend"),
    ("data engineer", "data engineering"), ("data science", "data science"),
    ("machine learning", "ml"), ("ml engineer", "ml"),
    ("devops", "devops"),
    ("mobile", "mobile"),
]


def _detect_role_subskills(query: str) -> list[str]:
    """If the query mentions a role type (e.g. 'backend engineer'), return representative
    skills for that role to use in skill_match scoring. Returns empty list if no role found."""
    lower = query.lower()
    for keyword, role_key in _ROLE_KEYWORDS:
        if keyword in lower:
            return _ROLE_SUBSKILLS.get(role_key, [])
    return []


def _match_skills_detail(
    required_names: list[str], profile_skills: list[Skill],
    raw_text: str | None = None, subskills: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[str]]:
    """Match skills using ONLY structured skills array.

    Returns (matched_names, missing_names) based on explicit skill entries.
    """
    matched: list[str] = []
    missing: list[str] = []
    _matcher = SkillMatcher(similarity_threshold=0.85)
    for rn in required_names:
        rn_subskills = subskills.get(rn) if subskills else None
        result = _matcher.find_best_match(rn, profile_skills, rn_subskills)
        if result is not None:
            matched.append(rn)
        else:
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
        self.scorer = scorer if scorer is not None else CandidateScorer()
        self.profile_store = profiles
        self._rerank_top_k = 20

    async def execute(
        self,
        parsed: ParsedQuery,
        top_k: int = 50,
        slider_weights: dict[str, float] | None = None,
        skip_reranker: bool = False,
    ) -> list[MatchResult]:
        search_text = self._query_to_search_text(parsed)

        # Check if there are active filters to decide retrieval size
        has_filters = False
        if parsed.location:
            if parsed.location.city and parsed.location.city.strip():
                has_filters = True
            if parsed.location.remote_ok:
                has_filters = True
        if parsed.experience:
            if parsed.experience.min_years is not None or parsed.experience.max_years is not None:
                has_filters = True
        if parsed.filters:
            if parsed.filters.exclude_companies or parsed.filters.include_companies:
                has_filters = True

        retrieval_k = max(1000, top_k * 10) if has_filters else top_k * 2

        query_vec = self.hybrid_search.embedder.embed_query(search_text)
        vector_raw = self.hybrid_search.vector_search.search(query_vec, top_k=retrieval_k)
        bm25_raw = self.hybrid_search.bm25_search.search(search_text, top_k=retrieval_k)

        hybrid_results = self.hybrid_search.reciprocal_rank_fusion(
            [vector_raw, bm25_raw], k=self.hybrid_search.rrf_k
        )

        vec_scores: dict[str, float] = {
            pid: self._norm_vec_score(s) for pid, s in vector_raw
        }
        bm25_scores: dict[str, float] = {
            pid: self._norm_bm25_score(s, bm25_raw) for pid, s in bm25_raw
        }

        fetch_top_k = max(self._rerank_top_k, top_k) if not skip_reranker else top_k * 2
        filtered = self._apply_filters(hybrid_results, parsed, limit=fetch_top_k)

        local_profile_cache: dict[str, Profile] = {}
        def _get_profile(pid: str) -> Profile | None:
            if pid in local_profile_cache:
                return local_profile_cache[pid]
            p = self.profile_store.get(pid)
            if p is not None:
                local_profile_cache[pid] = p
            return p

        pids_to_fetch = [pid for pid, _ in filtered[:fetch_top_k]]
        loop = asyncio.get_running_loop()
        num_fetch_workers = min(16, len(pids_to_fetch) or 1)
        with ThreadPoolExecutor(max_workers=num_fetch_workers) as fetch_pool:
            fetch_tasks = [
                loop.run_in_executor(fetch_pool, self.profile_store.get, pid)
                for pid in pids_to_fetch
            ]
            fetched_profiles = await asyncio.gather(*fetch_tasks)

        for pid, p in zip(pids_to_fetch, fetched_profiles):
            if p is not None:
                local_profile_cache[pid] = p

        if skip_reranker:
            candidate_scores = filtered[:top_k * 2]
            # Normalize RRF scores to [0, 1] so they work as cross_encoder_score dimension
            if candidate_scores:
                max_score = max(s for _, s in candidate_scores)
                if max_score > 0:
                    candidate_scores = [(pid, s / max_score) for pid, s in candidate_scores]
        else:
            rerank_candidates: list[tuple[str, str, float]] = []
            for pid, score in filtered[:fetch_top_k]:
                profile = local_profile_cache.get(pid)
                if profile is not None:
                    rerank_candidates.append((pid, profile.raw_text[:2000], score))
                else:
                    rerank_candidates.append((pid, "", score))
            candidate_scores = self.reranker.rerank(
                parsed.original_query or search_text, rerank_candidates, top_k=top_k
            )

        req_names = [rs.name for rs in parsed.required_skills]
        pref_names = [ps.name for ps in parsed.preferred_skills]
        all_req = req_names + pref_names

        scoring_args = []
        for pid, rerank_score in candidate_scores:
            profile = _get_profile(pid)
            if profile is None:
                continue
            scoring_args.append(
                (pid, rerank_score, profile, parsed, vec_scores, bm25_scores,
                 slider_weights, req_names, all_req, skip_reranker)
            )

        loop = asyncio.get_running_loop()
        num_workers = min(8, len(scoring_args) or 1)
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            tasks = [
                loop.run_in_executor(pool, self._score_single_candidate, *args)
                for args in scoring_args
            ]
            scored = await asyncio.gather(*tasks)
            results = [r for r in scored if r is not None]

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

    @staticmethod
    def _prepare_scores_dict(
        pid: str,
        profile: Profile,
        vec_scores: dict[str, float],
        bm25_scores: dict[str, float],
        rerank_score: float | None,
        skill_overlap: float,
        exp_match: float,
        all_req: list[str],
    ) -> dict[str, float | None]:
        """Build the raw scores dict for a single candidate, before weighting."""
        honeypot_reason = detect_honeypot(profile)
        honeypot_penalty = 0.15 if honeypot_reason else 1.0

        return {
            "semantic_similarity": vec_scores.get(pid),
            "keyword_match": bm25_scores.get(pid),
            "skill_match": skill_overlap * honeypot_penalty,
            "experience_match": exp_match * honeypot_penalty,
            "location_match": None,
            "education_match": None,
            "cross_encoder_score": rerank_score * honeypot_penalty if rerank_score else None,
            "behavioral_score": compute_behavioral_score(profile.signals) * honeypot_penalty,
            "career_trajectory_score": compute_career_trajectory(profile) * honeypot_penalty,
            "skill_proficiency_score": (
                compute_skill_proficiency(profile, all_req) * honeypot_penalty
            ),
        }

    @staticmethod
    def _extract_candidate_info(profile: Profile, pid: str) -> tuple[str, str | None, str | None, str | None, float | None]:  # noqa: E501
        """Extract basic candidate display info from a Profile."""
        loc = profile.personal.location
        return (
            profile.personal.name if profile.personal else "",
            profile.professional.current_title if profile.professional else None,
            profile.professional.current_company if profile.professional else None,
            loc.city if profile.personal and loc else None,
            profile.professional.total_experience_years if profile.professional else None,
        )

    def _score_single_candidate(
        self,
        pid: str,
        rerank_score: float | None,
        profile: Profile,
        parsed: ParsedQuery,
        vec_scores: dict[str, float],
        bm25_scores: dict[str, float],
        slider_weights: dict[str, float] | None,
        req_names: list[str],
        all_req: list[str],
        skip_reranker: bool = False,
    ) -> MatchResult | None:
        subskills = parsed.subskills if hasattr(parsed, "subskills") else None
        original_query = parsed.original_query or ""

        # When the query parser falls back to using the full original query as the
        # single required skill (e.g. "backend engineer with 5 years experience"),
        # no candidate's skill list will ever contain that phrase — skill_match
        # becomes 0 for everyone and loses all discriminating power.
        # Fix: detect the role type in the query and expand to representative skills
        # so backend engineers actually get credit for node.js, django, etc.
        role_subskills = _detect_role_subskills(original_query)
        is_fallback = (
            len(all_req) == 1
            and all_req[0].lower().strip() == original_query.lower().strip()
        )
        if role_subskills and is_fallback:
            # Replace the fallback phrase with actual role skills for matching
            effective_req = role_subskills
            effective_subskills: dict[str, list[str]] = {}
        else:
            effective_req = list(all_req)
            effective_subskills = dict(subskills) if subskills else {}

        matched_skills_list, missing_skills_list = _match_skills_detail(
            effective_req, profile.skills, profile.raw_text, effective_subskills,
        )
        skill_overlap = len(matched_skills_list) / len(effective_req) if effective_req else 1.0

        total_years = (
            profile.professional.total_experience_years
            if profile.professional and profile.professional.total_experience_years
            else 0
        )

        from src.matching.experience_matcher import ExperienceMatcher
        exp_matcher = ExperienceMatcher()
        years_match = exp_matcher.match(
            required_min_years=parsed.experience.min_years,
            required_max_years=parsed.experience.max_years,
            candidate_years=total_years,
        )
        title_match = exp_matcher.match_title(parsed.original_query or "", profile)

        exp_match = min(1.0, total_years / 10.0) * years_match

        scores_dict = ExecutorAgent._prepare_scores_dict(
            pid, profile, vec_scores, bm25_scores, rerank_score,
            skill_overlap, exp_match, list(effective_req),
        )

        match_scores = self.scorer.compute_overall(scores_dict, slider_weights)
        match_scores.overall = max(0.0, min(1.0, match_scores.overall * title_match))

        # For display: show the actual backend/frontend skills that were matched/missed
        if role_subskills and is_fallback:
            req_only_matched = list(matched_skills_list)
            req_only_missing = []  # Don't show the full-query phrase as a "missing skill"
        else:
            req_only_matched_l, req_only_missing_l = _match_skills_detail(
                req_names, profile.skills, profile.raw_text, dict(subskills) if subskills else {},
            )
            req_only_matched = req_only_matched_l
            req_only_missing = req_only_missing_l

        name, title, company, city, exp_years = ExecutorAgent._extract_candidate_info(profile, pid)

        return MatchResult(
            query_id="",
            profile_id=pid,
            rank=0,
            name=name,
            current_title=title,
            current_company=company,
            location=city,
            experience_years=exp_years,
            scores=match_scores,
            matched_skills=list(set(req_only_matched)),
            missing_skills=list(set(req_only_missing)),
            metadata=MatchMetadata(search_method=SearchMethod.HYBRID, reranked=True),
        )

    def _query_to_search_text(self, parsed: ParsedQuery) -> str:
        # Prefer original_query for retrieval: it's more natural for BM25/vector
        # and avoids "skill fallback" strings like "backend engineer with 5 years experience"
        # being broken into tokens that match the wrong candidates.
        if parsed.original_query and parsed.original_query.strip():
            base = parsed.original_query.strip()
            # Append location if present so location-scoped queries still work
            if parsed.location and parsed.location.city:
                base = f"{base} {parsed.location.city}"
            return base

        # Fallback: reconstruct from parsed skill names
        parts: list[str] = []
        for rs in parsed.required_skills:
            parts.append(rs.name)
        for ps in parsed.preferred_skills:
            parts.append(ps.name)
        if parsed.location and parsed.location.city:
            parts.append(parsed.location.city)
        return " ".join(parts) if parts else "software engineer"

    def _apply_filters(
        self,
        results: list[tuple[str, float]],
        parsed: ParsedQuery,
        limit: int | None = None,
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

            if limit is not None and len(filtered) >= limit:
                break

        return filtered
