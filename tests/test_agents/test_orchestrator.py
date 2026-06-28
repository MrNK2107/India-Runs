from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.orchestrator import Orchestrator
from src.agents.planner import PlannerAgent
from src.agents.reflector import ReflectorAgent
from src.agents.executor import ExecutorAgent
from src.core.models import ParsedQuery, LocationRequirements, ExperienceRequirements, QueryFilters


@pytest.mark.asyncio
async def test_orchestrator_replan_triggered():
    # Setup mocks for agents
    planner = MagicMock(spec=PlannerAgent)
    planner.replan = AsyncMock(return_value=ParsedQuery(
        original_query="Find Python in Bangalore",
        location=LocationRequirements(city="Bangalore"),
    ))
    
    executor = MagicMock(spec=ExecutorAgent)
    executor.execute = AsyncMock(return_value=[])
    
    reflector = MagicMock(spec=ReflectorAgent)
    reflector.reflect = AsyncMock(return_value={
        "should_replan": True,
        "feedback": "No candidates found.",
    })
    
    orchestrator = Orchestrator(planner=planner, executor=executor, reflector=reflector)
    
    # State with raw_query and initial parsed_query (similar to search from UI)
    state = {
        "raw_query": "Find Python in Bangalore",
        "parsed_query": {
            "required_skills": [{"name": "Python", "importance": "required"}],
            "preferred_skills": [],
            "subskills": {},
            "experience": {"min_years": 5, "max_years": None},
            "location": {"city": "Bangalore", "remote_ok": False},
            "education": {"min_degree": None, "field": None},
            "salary": {"min": None, "max": None, "currency": "INR"},
            "filters": {"exclude_companies": [], "include_companies": []},
            "original_query": "Find Python in Bangalore",
        },
        "results": [],
        "evaluations": {
            "should_replan": True,
            "feedback": "No candidates found.",
        },
        "replan_count": 1,
        "max_replans": 3,
        "should_continue": True,
        "start_time_ms": 0,
        "slider_weights": {},
        "listwise_ranked": False,
        "top_k": 10,
        "filters": None,
        "rationales": None,
    }
    
    # Run the _plan_node method
    res = await orchestrator._plan_node(state)
    
    # Assert that replan was actually called on the planner (which was bypassed previously)
    planner.replan.assert_called_once()
    assert res["parsed_query"]["location"]["city"] == "Bangalore"


@pytest.mark.asyncio
async def test_executor_retrieval_k_with_filters():
    # Setup mocks
    hybrid = MagicMock()
    hybrid.embedder = MagicMock()
    hybrid.embedder.embed_query = MagicMock(return_value=None)
    hybrid.vector_search = MagicMock()
    hybrid.vector_search.search = MagicMock(return_value=[])
    hybrid.bm25_search = MagicMock()
    hybrid.bm25_search.search = MagicMock(return_value=[])
    hybrid.reciprocal_rank_fusion = MagicMock(return_value=[])
    
    reranker = MagicMock()
    scorer = MagicMock()
    profile_store = MagicMock()
    
    executor = ExecutorAgent(hybrid, reranker, scorer, profile_store)
    
    # Query with filters
    parsed_with_filters = ParsedQuery(
        location=LocationRequirements(city="Bangalore"),
        experience=ExperienceRequirements(min_years=5.0),
    )
    
    await executor.execute(parsed_with_filters, top_k=10)
    
    # Verify retrieval size is increased to 1000
    hybrid.vector_search.search.assert_called_with(None, top_k=1000)
    hybrid.bm25_search.search.assert_called_with("5+ years experience Bangalore", top_k=1000)
    
    # Query without filters
    hybrid.vector_search.search.reset_mock()
    hybrid.bm25_search.reset_mock()
    parsed_no_filters = ParsedQuery()
    
    await executor.execute(parsed_no_filters, top_k=10)
    
    # Verify retrieval size is top_k * 2 (20)
    hybrid.vector_search.search.assert_called_with(None, top_k=20)
    hybrid.bm25_search.search.assert_called_with("software engineer", top_k=20)
