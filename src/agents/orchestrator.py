from __future__ import annotations

import logging
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agents.executor import ExecutorAgent
from src.agents.planner import PlannerAgent
from src.agents.reflector import ReflectorAgent
from src.core.config import get_scoring_config
from src.core.models import (
    MatchResult,
    ParsedQuery,
    SearchMetadata,
    SearchResponse,
    SearchResultItem,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    raw_query: str
    parsed_query: dict | None
    results: list[dict]
    evaluations: dict | None
    replan_count: int
    max_replans: int
    should_continue: bool
    search_metadata: dict
    total_candidates_searched: int
    start_time_ms: int


class Orchestrator:
    def __init__(
        self, planner: PlannerAgent, executor: ExecutorAgent, reflector: ReflectorAgent,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.reflector = reflector
        config = get_scoring_config()
        self.max_replans = config.get("max_replan_cycles", 3)
        self.min_good_matches = config.get("min_good_matches_for_pass", 8)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("plan", self._plan_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("generate_rationale", self._rationale_node)

        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "execute")
        workflow.add_edge("execute", "reflect")
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue,
            {
                "replan": "plan",
                "done": "generate_rationale",
            },
        )
        workflow.add_edge("generate_rationale", END)

        return workflow.compile()

    async def run(self, raw_query: str) -> SearchResponse:
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
        }
        final_state = await self.graph.ainvoke(initial_state)
        return self._build_response(final_state)

    async def _plan_node(self, state: AgentState) -> dict:
        query = state["raw_query"]
        if state["replan_count"] > 0 and state["evaluations"]:
            feedback = ""
            if isinstance(state["evaluations"], dict):
                feedback = state["evaluations"].get("feedback", "")
            previous_params = state.get("parsed_query") or {}
            parsed = await self.planner.replan(query, previous_params, feedback)
        else:
            parsed = await self.planner.plan(query)

        parsed_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else {}
        return {"parsed_query": parsed_dict}

    async def _execute_node(self, state: AgentState) -> dict:
        parsed_dict = state.get("parsed_query") or {}
        parsed = ParsedQuery(**parsed_dict) if parsed_dict else ParsedQuery()
        results = await self.executor.execute(parsed, top_k=50)
        methods_used = []
        if state.get("replan_count", 0) > 0:
            methods_used.append("hybrid+rerank+replan")
        else:
            methods_used.append("hybrid+rerank")

        return {
            "results": [r.model_dump() for r in results],
            "total_candidates_searched": len(results),
            "search_metadata": {
                "methods_used": methods_used,
                "replan_count": state.get("replan_count", 0),
            },
        }

    async def _reflect_node(self, state: AgentState) -> dict:
        parsed_dict = state.get("parsed_query") or {}
        parsed = ParsedQuery(**parsed_dict) if parsed_dict else ParsedQuery()
        results = [MatchResult(**r) for r in state.get("results", [])]
        evaluations = await self.reflector.reflect(parsed, results)
        return {"evaluations": evaluations}

    def _should_continue(self, state: AgentState) -> str:
        evaluations = state.get("evaluations", {})
        should_replan = False
        if isinstance(evaluations, dict):
            should_replan = evaluations.get("should_replan", False)

        if should_replan and state["replan_count"] < state["max_replans"]:
            return "replan"
        return "done"

    async def _rationale_node(self, state: AgentState) -> dict:
        return {"should_continue": False}

    def _build_response(self, state: AgentState) -> SearchResponse:
        results_raw = state.get("results", [])
        items: list[SearchResultItem] = []

        for i, r in enumerate(results_raw[:100], start=1):
            if isinstance(r, dict):
                items.append(
                    SearchResultItem(
                        rank=i,
                        profile_id=r.get("profile_id", ""),
                        name=r.get("name", ""),
                        current_title=r.get("current_title"),
                        current_company=r.get("current_company"),
                        location=r.get("location"),
                        experience_years=r.get("experience_years"),
                        scores=r.get("scores", {}),
                        matched_skills=r.get("matched_skills", []),
                        missing_skills=r.get("missing_skills", []),
                    )
                )

        total_time = int(time.time() * 1000) - state.get("start_time_ms", 0)
        metadata = SearchMetadata(
            methods_used=state.get("search_metadata", {}).get("methods_used", []),
            replan_count=state.get("replan_count", 0),
            total_time_ms=total_time,
        )

        return SearchResponse(
            query_id="",
            total_candidates_searched=state.get("total_candidates_searched", 0),
            results=items,
            processing_time_ms=total_time,
            search_metadata=metadata,
        )
