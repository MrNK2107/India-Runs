from __future__ import annotations

import logging
import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.executor import ExecutorAgent
from src.agents.planner import PlannerAgent
from src.agents.reflector import ReflectorAgent
from src.core.config import get_scoring_config, get_settings
from src.core.models import (
    ExperienceRequirements,
    LocationRequirements,
    MatchResult,
    ParsedQuery,
    QueryFilters,
    Rationale,
    SearchFilters,
    SearchMetadata,
    SearchResponse,
    SearchResultItem,
)
from src.ranking.listwise_ranker import PlackettLuceRanker
from src.rationale.generator import RationaleGenerator

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    raw_query: str
    parsed_query: dict[str, Any] | None
    results: list[dict[str, Any]]
    evaluations: dict[str, Any] | None
    replan_count: int
    max_replans: int
    should_continue: bool
    search_metadata: dict[str, Any]
    total_candidates_searched: int
    start_time_ms: int
    slider_weights: dict[str, float]
    listwise_ranked: bool
    top_k: int
    filters: dict[str, Any] | None


_SIMPLE_QUERY_RE = re.compile(r"^(find|search|get|show|i need|looking for|need)\s+", re.IGNORECASE)


def _is_simple_query(query: str) -> bool:
    """Detect queries that can skip LLM parsing and use rule-based extraction."""
    words = query.strip().split()
    if len(words) <= 3:
        return True
    # Queries matching skill/location patterns known by the rule-based parser
    if _SIMPLE_QUERY_RE.match(query):
        return len(words) <= 4
    # Queries with 4-6 words that look like skill+role+location combos
    # e.g. "Python developer AWS Bangalore", "Java spring boot Mumbai"
    if len(words) <= 6:
        from src.core.constants import INDIAN_CITIES
        lower = query.strip().lower()
        # Check if it contains a known Indian city
        has_city = any(c.lower() in lower for c in INDIAN_CITIES)
        # Check if it looks like a tech skills query (common tech terms)
        tech_keywords = {"python", "java", "aws", "react", "node", "javascript", "typescript",
                         "golang", "rust", "sql", "docker", "kubernetes", "devops", "ml",
                         "machine learning", "data", "full stack", "backend", "frontend",
                         "sde", "software", "engineering", "developer", "engineer", "analyst",
                         "manager", "architect", "lead", "intern", "fresher"}
        has_tech = any(kw in lower for kw in tech_keywords)
        return has_city or has_tech
    return False


STOP_WORDS = frozenset({
    "find", "search", "get", "show", "need", "looking", "for", "with",
    "and", "the", "a", "an", "in", "at", "on", "to", "of", "is", "are",
    "i", "we", "you", "he", "she", "it", "they", "me", "my", "our",
    "senior", "junior", "lead", "head", "principal", "staff",
    "developer", "engineer", "manager", "architect", "analyst",
    "experience", "role", "position", "job", "opening", "hire",
    "dev", "sr", "jr", "intern", "fresher",
    # Generic descriptors — not real skills
    "engineering", "tech", "technology", "technical",
    "management", "team", "agile", "leadership",
    "software", "full", "stack", "backend", "frontend", "back end", "front end",
    "cloud", "infrastructure", "platform", "system", "systems",
    "design", "development", "testing", "deployment",
    "solution", "solutions", "application", "applications",
    "digital", "transformation", "innovation",
    "strategy", "strategic", "planning", "operations",
    "product", "program", "project", "portfolio",
})


def _dict_to_match_result(r: dict, rank_override: int | None = None) -> MatchResult:
    """Convert a raw results dict (from state or JSON) into a MatchResult.

    Extracted to eliminate 4x-repeated reconstruction blocks across
    _listwise_ranking_node, _rationale_node, _build_response, and _turbo_run.
    """
    from src.core.models import MatchScores

    scores_dict = r.get("scores", {})
    if not isinstance(scores_dict, dict):
        scores_dict = {}
    match_scores = MatchScores(**scores_dict) if scores_dict else MatchScores()

    return MatchResult(
        query_id=r.get("query_id", ""),
        profile_id=r.get("profile_id", ""),
        rank=rank_override if rank_override is not None else r.get("rank", 1),
        name=r.get("name", ""),
        current_title=r.get("current_title"),
        current_company=r.get("current_company"),
        location=r.get("location"),
        experience_years=r.get("experience_years"),
        scores=match_scores,
        matched_skills=r.get("matched_skills", []),
        missing_skills=r.get("missing_skills", []),
    )


def _merge_filter_list(
    existing: list[str], incoming: list[str] | None,
) -> list[str]:
    """Merge two filter lists (deduped), handling None incoming."""
    if not incoming:
        return list(existing)
    return list(set(existing + incoming))


def _parse_query_text(text: str) -> ParsedQuery:
    from src.core.constants import INDIAN_CITIES
    from src.core.models import RequiredSkill

    lower = text.strip().lower()

    city = None
    for c in sorted(INDIAN_CITIES, key=len, reverse=True):
        if c.lower() in lower:
            city = c
            break

    words = lower.split()
    skill_words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    # Exclude detected city name from skills
    if city:
        skill_words = [w for w in skill_words if w != city.lower()]
    deduped = list(dict.fromkeys(skill_words))

    return ParsedQuery(
        required_skills=[RequiredSkill(name=s) for s in deduped[:10]],
        experience=ExperienceRequirements(),
        location=LocationRequirements(city=city),
        filters=QueryFilters(),
        original_query=text.strip(),
    )


class Orchestrator:
    def __init__(
        self, planner: PlannerAgent, executor: ExecutorAgent, reflector: ReflectorAgent,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.reflector = reflector
        self.rationale_gen = RationaleGenerator()
        self.listwise_ranker = PlackettLuceRanker()
        config = get_scoring_config()
        settings = get_settings()
        self.max_replans = settings.max_replan_cycles
        self.min_good_matches = config.get("min_good_matches_for_pass", 8)
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()


    def _build_graph(self) -> StateGraph[AgentState]:
        workflow = StateGraph(AgentState)

        workflow.add_node("plan", self._plan_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("listwise_ranking", self._listwise_ranking_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("generate_rationale", self._rationale_node)

        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "execute")
        workflow.add_edge("execute", "listwise_ranking")
        workflow.add_edge("listwise_ranking", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue,
            {
                "replan": "plan",
                "done": "generate_rationale",
            },
        )
        workflow.add_edge("generate_rationale", END)

        return workflow

    async def run(
        self, raw_query: str,
        slider_weights: dict[str, float] | None = None,
        use_turbo: bool = False,
        top_k: int = 50,
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        if use_turbo:
            return await self._turbo_run(raw_query, slider_weights, top_k, filters)

        initial_state: AgentState = {
            "raw_query": raw_query,
            "parsed_query": None,
            "results": [],
            "evaluations": None,
            "replan_count": 0,
            "max_replans": self.max_replans,
            "should_continue": True,
            "search_metadata": {},
            "total_candidates_searched": 0,
            "start_time_ms": int(time.time() * 1000),
            "slider_weights": slider_weights or {},
            "listwise_ranked": False,
            "top_k": top_k,
            "filters": filters.model_dump() if filters else None,
        }
        final_state: AgentState = await self.compiled_graph.ainvoke(initial_state)  # type: ignore[assignment]
        return self._build_response(final_state)

    async def _turbo_run(
        self, raw_query: str,
        slider_weights: dict[str, float] | None = None,
        top_k: int = 50,
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        parsed = _parse_query_text(raw_query)
        if filters:
            if filters.location:
                parsed.location.city = filters.location
            if filters.min_experience_years is not None:
                parsed.experience.min_years = filters.min_experience_years
            if filters.max_experience_years is not None:
                parsed.experience.max_years = filters.max_experience_years
            parsed.location.remote_ok = filters.remote_ok or parsed.location.remote_ok
            parsed.filters.exclude_companies = _merge_filter_list(
                parsed.filters.exclude_companies, filters.exclude_companies,
            )
            parsed.filters.include_companies = _merge_filter_list(
                parsed.filters.include_companies, filters.include_companies,
            )

        start_time = int(time.time() * 1000)
        results = await self.executor.execute(parsed, top_k=top_k, slider_weights=slider_weights)
        elapsed = int(time.time() * 1000) - start_time
        items: list[SearchResultItem] = []
        for i, r in enumerate(results[:100], start=1):
            rationale = self.rationale_gen._template_rationale(r, None)
            items.append(
                SearchResultItem(
                    rank=i,
                    profile_id=r.profile_id,
                    name=r.name,
                    current_title=r.current_title,
                    current_company=r.current_company,
                    location=r.location,
                    experience_years=r.experience_years,
                    scores=r.scores,
                    matched_skills=r.matched_skills,
                    missing_skills=r.missing_skills,
                    rationale=Rationale(
                        summary=rationale.summary,
                        strengths=rationale.strengths,
                        gaps=rationale.gaps,
                        skill_details=rationale.skill_details,
                        experience_analysis=rationale.experience_analysis,
                        recommendation=rationale.recommendation,
                    ),
                )
            )
        return SearchResponse(
            query_id="",
            total_candidates_searched=len(results),
            results=items,
            processing_time_ms=elapsed,
            search_metadata=SearchMetadata(
                methods_used=["turbo"],
                replan_count=0,
                total_time_ms=elapsed,
                listwise_ranked=False,
            ),
        )

    async def _plan_node(self, state: AgentState    ) -> dict[str, Any]:
        query = state["raw_query"]
        if state["replan_count"] > 0 and state["evaluations"]:
            feedback = ""
            if isinstance(state["evaluations"], dict):
                feedback = state["evaluations"].get("feedback", "")
            previous_params = state.get("parsed_query") or {}
            parsed = await self.planner.replan(query, previous_params, feedback)
        else:
            parsed = await self.planner.plan(query)

        # Merge filters from state if present
        f = state.get("filters")
        if f:
            if f.get("location"):
                parsed.location.city = f["location"]
            if f.get("min_experience_years") is not None:
                parsed.experience.min_years = float(f["min_experience_years"])
            if f.get("max_experience_years") is not None:
                parsed.experience.max_years = float(f["max_experience_years"])
            if f.get("remote_ok"):
                parsed.location.remote_ok = bool(f["remote_ok"])
            exclude = f.get("exclude_companies") or None
            parsed.filters.exclude_companies = _merge_filter_list(
                parsed.filters.exclude_companies, exclude,
            )
            include = f.get("include_companies") or None
            parsed.filters.include_companies = _merge_filter_list(
                parsed.filters.include_companies, include,
            )

        parsed_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else {}
        return {"parsed_query": parsed_dict}

    async def _execute_node(self, state: AgentState) -> dict[str, Any]:
        parsed_dict = state.get("parsed_query") or {}
        parsed = (
            ParsedQuery.model_validate(parsed_dict, strict=False)
            if parsed_dict else ParsedQuery()
        )
        slider_weights = state.get("slider_weights") or None
        top_k = state.get("top_k", 50)
        results = await self.executor.execute(parsed, top_k=top_k, slider_weights=slider_weights)
        methods_used = []
        if state.get("replan_count", 0) > 0:
            methods_used.append("hybrid+rerank+replan")
        else:
            methods_used.append("hybrid+rerank")

        return {
            "results": [r.model_dump() for r in results],
            "total_candidates_searched": len(results),
            "listwise_ranked": False,
            "search_metadata": {
                "methods_used": methods_used,
                "replan_count": state.get("replan_count", 0),
            },
        }

    async def _listwise_ranking_node(self, state: AgentState) -> dict[str, Any]:
        results_raw = state.get("results", [])
        if not results_raw or len(results_raw) < 2:
            return {"listwise_ranked": False}

        from src.fairness.anonymizer import anonymize_profile

        match_results = []
        anonymized_profiles: dict[str, dict] = {}
        for r in results_raw:
            if isinstance(r, dict):
                match_results.append(_dict_to_match_result(r))

            pid = r.get("profile_id", "") if isinstance(r, dict) else r.profile_id
            if pid and hasattr(self.executor, "profile_store"):
                profile = self.executor.profile_store.get(pid)
                if profile is not None:
                    anonymized_profiles[pid] = anonymize_profile(profile)

        if len(match_results) < 2:
            return {"listwise_ranked": False}

        take_top = min(20, len(match_results))
        top_candidates = match_results[:take_top]

        ranked = await self.listwise_ranker.arank(
            top_candidates, anonymized_profiles=anonymized_profiles
        )
        rank_map = {pid: merit for pid, merit in ranked}

        ordered_results = sorted(
            results_raw,
            key=lambda r: rank_map.get(r.get("profile_id", ""), 0),
            reverse=True,
        )

        methods = state.get("search_metadata", {})
        if isinstance(methods, dict):
            ml = methods.get("methods_used", [])
            if "listwise_ranking" not in ml:
                ml.append("listwise_ranking")
                methods["methods_used"] = ml

        return {
            "results": ordered_results,
            "listwise_ranked": True,
            "search_metadata": methods,
        }

    async def _reflect_node(self, state: AgentState) -> dict[str, Any]:
        parsed_dict = state.get("parsed_query") or {}
        parsed = (
            ParsedQuery.model_validate(parsed_dict, strict=False)
            if parsed_dict else ParsedQuery()
        )
        results = [MatchResult(**r) for r in state.get("results", [])]
        evaluations = await self.reflector.reflect(parsed, results)
        replan_count = state.get("replan_count", 0)
        ev = evaluations if isinstance(evaluations, dict) else {}
        should_replan = ev.get("should_replan", False)
        if should_replan:
            replan_count += 1
        return {"evaluations": evaluations, "replan_count": replan_count}

    def _should_continue(self, state: AgentState) -> str:
        evaluations = state.get("evaluations", {})
        should_replan = False
        if isinstance(evaluations, dict):
            should_replan = evaluations.get("should_replan", False)

        if should_replan and state["replan_count"] < state["max_replans"]:
            return "replan"
        return "done"

    async def _rationale_node(self, state: AgentState) -> dict[str, Any]:
        results_raw = state.get("results", [])
        if not results_raw:
            return {"should_continue": False}

        rationales = []
        for r in results_raw[:20]:
            if isinstance(r, dict):
                match_result = _dict_to_match_result(r)
                rationale = self.rationale_gen._template_rationale(match_result, None)
                rationales.append({
                    "profile_id": r.get("profile_id", ""),
                    "rationale": rationale.model_dump() if hasattr(rationale, "model_dump") else {},
                })

        return {"should_continue": False, "rationales": rationales}

    def _build_response(self, state: AgentState) -> SearchResponse:
        results_raw = state.get("results", [])
        items: list[SearchResultItem] = []

        for i, r in enumerate(results_raw[:100], start=1):
            if isinstance(r, dict):
                match_result = _dict_to_match_result(r, rank_override=i)
                rationale = self.rationale_gen._template_rationale(match_result, None)
                items.append(
                    SearchResultItem(
                        rank=i,
                        profile_id=match_result.profile_id,
                        name=match_result.name,
                        current_title=match_result.current_title,
                        current_company=match_result.current_company,
                        location=match_result.location,
                        experience_years=match_result.experience_years,
                        scores=match_result.scores,
                        matched_skills=match_result.matched_skills,
                        missing_skills=match_result.missing_skills,
                        rationale=Rationale(
                            summary=rationale.summary,
                            strengths=rationale.strengths,
                            gaps=rationale.gaps,
                            skill_details=rationale.skill_details,
                            experience_analysis=rationale.experience_analysis,
                            recommendation=rationale.recommendation,
                        ),
                    )
                )

        total_time = int(time.time() * 1000) - state.get("start_time_ms", 0)
        metadata = SearchMetadata(
            methods_used=state.get("search_metadata", {}).get("methods_used", []),
            replan_count=state.get("replan_count", 0),
            total_time_ms=total_time,
            listwise_ranked=state.get("listwise_ranked", False),
        )

        return SearchResponse(
            query_id="",
            total_candidates_searched=state.get("total_candidates_searched", 0),
            results=items,
            processing_time_ms=total_time,
            search_metadata=metadata,
        )
