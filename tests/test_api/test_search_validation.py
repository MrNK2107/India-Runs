"""Edge-case validation for POST /api/v1/search: clear 422s instead of silent misbehaviour."""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.api.routes.search as search_route
from src.core.models import SLIDER_DIMENSIONS, SearchFilters, SearchRequest
from src.matching.scorer import DIM_TO_ACTUAL
from tests.test_api.test_routes import FakeOrchestrator


@pytest.fixture
def api(client: TestClient) -> tuple[TestClient, FakeOrchestrator]:
    fake = FakeOrchestrator()
    search_route.init_orchestrator(fake)  # type: ignore[arg-type]
    return client, fake


def _post(api: tuple[TestClient, FakeOrchestrator], payload: dict[str, Any]) -> Any:
    client, _ = api
    return client.post("/api/v1/search", json=payload)


def _msg(r: Any) -> str:
    return " ".join(str(e.get("msg", "")) for e in r.json()["detail"])


def test_slider_dimensions_match_scorer() -> None:
    assert set(SLIDER_DIMENSIONS) == set(DIM_TO_ACTUAL)


@pytest.mark.parametrize("query", ["   ", "\n\t", "　"])
def test_blank_query_is_422(api, query: str) -> None:
    r = _post(api, {"query": query})
    assert r.status_code == 422
    assert "blank" in _msg(r)
    assert api[1].calls == []


def test_query_is_stripped_before_forwarding(api) -> None:
    r = _post(api, {"query": "  python developer \n"})
    assert r.status_code == 200
    assert api[1].calls[0]["raw_query"] == "python developer"


def test_unicode_query_is_accepted(api) -> None:
    r = _post(api, {"query": "पायथन डेवलपर बेंगलुरु"})
    assert r.status_code == 200
    assert api[1].calls[0]["raw_query"] == "पायथन डेवलपर बेंगलुरु"


def test_unknown_slider_key_is_422_and_lists_valid_keys(api) -> None:
    # Previously {"skills": 70} was accepted and silently ignored.
    r = _post(api, {"query": "python", "slider_weights": {"skills": 70}})
    assert r.status_code == 422
    msg = _msg(r)
    assert "skills" in msg and "skill_match" in msg
    assert api[1].calls == []


@pytest.mark.parametrize("value", [-1, 100.5, 1e308])
def test_out_of_range_slider_value_is_422(api, value: float) -> None:
    r = _post(api, {"query": "python", "slider_weights": {"skill_match": value}})
    assert r.status_code == 422
    assert "0 and 100" in _msg(r)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_slider_value_is_422(api, token: str) -> None:
    # Python's json accepts NaN/Infinity; an infinite weight made every overall score 1.0.
    client, fake = api
    body = '{"query": "python", "slider_weights": {"skill_match": ' + token + '}}'
    r = client.post("/api/v1/search", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert fake.calls == []


def test_valid_slider_weights_forwarded(api) -> None:
    weights = {"skill_match": 50, "experience_match": 0, "cultural_fit": 100}
    r = _post(api, {"query": "python", "slider_weights": weights})
    assert r.status_code == 200
    assert api[1].calls[0]["slider_weights"] == weights


@pytest.mark.parametrize(
    "filters,needle",
    [
        ({"min_experience_years": 10, "max_experience_years": 3}, "must not exceed"),
        ({"min_experience_years": -1}, "non-negative"),
        ({"max_experience_years": -0.5}, "non-negative"),
    ],
)
def test_bad_experience_filters_are_422(api, filters: dict, needle: str) -> None:
    r = _post(api, {"query": "python", "filters": filters})
    assert r.status_code == 422
    assert needle in _msg(r)


def test_equal_min_max_experience_is_ok(api) -> None:
    r = _post(api, {"query": "python",
                    "filters": {"min_experience_years": 5, "max_experience_years": 5}})
    assert r.status_code == 200


def test_unknown_top_level_and_filter_fields_are_ignored(api) -> None:
    r = _post(api, {"query": "python", "foo": 1, "filters": {"bar": 2}})
    assert r.status_code == 200


def test_null_filters_and_wrong_types_are_422(api) -> None:
    assert _post(api, {"query": "python", "filters": None}).status_code == 422
    assert _post(api, {"query": "python", "filters": []}).status_code == 422
    assert _post(api, {"query": ["python"]}).status_code == 422


def test_json_array_body_is_422(api) -> None:
    client, fake = api
    r = client.post("/api/v1/search", json=[{"query": "python"}])
    assert r.status_code == 422
    assert fake.calls == []


def test_internal_search_filters_stay_permissive() -> None:
    # The executor builds SearchFilters from LLM-parsed queries; inconsistent values there
    # must not raise (only the API request is strict).
    SearchFilters(min_experience_years=10, max_experience_years=2)
    SearchRequest(query="python")


@pytest.mark.parametrize(
    "body",
    [
        '{"query": "python", "max_results": NaN}',
        '{"query": "python", "filters": {"min_experience_years": Infinity}}',
    ],
)
def test_non_finite_input_anywhere_gives_422_not_500(api, body: str) -> None:
    # Regression: the default 422 body echoed NaN/Infinity and failed to render -> 500.
    client, _ = api
    r = client.post("/api/v1/search", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], list)
