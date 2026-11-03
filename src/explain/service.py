"""Glue between the search pipeline and the explainer (used by the API routes)."""

from __future__ import annotations

from typing import Any

from src.core.models import (
    Explanation,
    MatchScores,
    ParsedQuery,
    SearchFilters,
    SearchResponse,
)
from src.core.query_parser import parse_query
from src.explain.evidence import explain_candidate

_SCORE_DIMS = (
    "semantic_similarity", "keyword_match", "skill_match", "experience_match",
    "location_match", "education_match", "cross_encoder_score", "behavioral_score",
    "career_trajectory_score", "skill_proficiency_score",
)


def parsed_for(query: str, filters: SearchFilters | None = None) -> ParsedQuery:
    parsed = parse_query(query)
    if filters is not None:
        if filters.location:
            parsed.location.city = filters.location
        if filters.min_experience_years is not None:
            parsed.experience.min_years = filters.min_experience_years
        if filters.max_experience_years is not None:
            parsed.experience.max_years = filters.max_experience_years
    return parsed


def scores_to_dict(scores: MatchScores) -> dict[str, float | None]:
    out: dict[str, float | None] = {d: getattr(scores, d) for d in _SCORE_DIMS}
    # MatchScores stores "not retrieved by this channel" as 0.0 for these two, but the
    # scorer skipped them (None), weight included. A real vector score is (cos+1)/2 and
    # never exactly 0, and a BM25 hit with score 0 is not a hit, so 0.0 means absent.
    for dim in ("semantic_similarity", "keyword_match"):
        if out[dim] == 0.0:
            out[dim] = None
    return out


def attach_explanations(
    response: SearchResponse,
    executor: Any,
    query: str,
    filters: SearchFilters | None,
    slider_weights: dict[str, float] | None,
    include_current_tenure: bool,
    credit_work_history: bool = False,
) -> SearchResponse:
    parsed = parsed_for(query, filters)
    weights = executor.scorer.effective_weights(slider_weights or None)
    for item in response.results:
        profile = executor.profile_store.get(item.profile_id)
        if profile is None:
            continue
        item.explanation = explain_candidate(
            profile, parsed, scores_to_dict(item.scores), item.scores.overall, weights,
            include_current_tenure=include_current_tenure,
            credit_work_history=credit_work_history,
        )
    return response


def explain_profile(
    executor: Any,
    query: str,
    profile_id: str,
    slider_weights: dict[str, float] | None,
    include_current_tenure: bool,
    credit_work_history: bool = False,
) -> Explanation | None:
    """Score and explain one candidate for a query, whether or not it made the list
    ("why isn't X ranked?"). Uses the same scoring path as search, with the vector
    similarity computed directly, keyword overlap looked up in the BM25 index, and no
    rerank component (that one is relative to the rest of a result list)."""
    profile = executor.profile_store.get(profile_id)
    if profile is None:
        return None
    parsed = parsed_for(query)
    embedder = executor.hybrid_search.embedder
    q = embedder.embed_query(parsed.original_query or query)
    p = embedder.embed(profile.raw_text or "")
    cos = float(sum(float(a) * float(b) for a, b in zip(q, p)))
    vec_scores = {profile_id: max(0.0, min(1.0, (cos + 1.0) / 2.0))}
    # Keyword overlap on the same scale as search (relative to the best BM25 hit);
    # outside the top 1000 keyword hits it counts as 0.
    bm25_scores: dict[str, float] = {}
    try:
        hits = executor.hybrid_search.bm25_search.search(parsed.original_query or query, top_k=1000)
    except Exception:  # noqa: BLE001 - keyword index optional for an explanation
        hits = []
    best = max((sc for _, sc in hits), default=0.0)
    for pid, sc in hits:
        if pid == profile_id and best > 0:
            bm25_scores[pid] = max(0.0, min(1.0, sc / best))
    req_names = [rs.name for rs in parsed.required_skills]
    all_req = req_names + [ps.name for ps in parsed.preferred_skills]
    result = executor._score_single_candidate(
        profile_id, None, profile, parsed, vec_scores, bm25_scores, slider_weights or None,
        req_names, all_req, True, include_current_tenure, credit_work_history,
    )
    if result is None:
        return None
    weights = executor.scorer.effective_weights(slider_weights or None)
    return explain_candidate(
        profile, parsed, scores_to_dict(result.scores), result.scores.overall, weights,
        include_current_tenure=include_current_tenure, credit_work_history=credit_work_history,
    )
