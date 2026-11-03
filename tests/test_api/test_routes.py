"""HTTP-level tests for the FastAPI app: routes, middleware, lifespan.

No models, indexes or LLMs are loaded: the orchestrator and profile store are
replaced with in-memory fakes, and the lifespan runs against an empty tmp dir.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.api.routes.ingest as ingest_route
import src.api.routes.profiles as profiles_route
import src.api.routes.search as search_route
import src.main as main_module
from src.api.routes.health import init_health, set_index_size, set_model_loaded
from src.core.models import (
    PersonalInfo,
    Profile,
    ProfileSource,
    SearchFilters,
    SearchResponse,
)


class FakeOrchestrator:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = fail

    async def run(self, **kwargs: Any) -> SearchResponse:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("secret internal path C:/models/x.bin missing")
        return SearchResponse(query_id="q-1", total_candidates_searched=42)


class FakeProfileStore:
    def __init__(self, profiles: list[Profile]) -> None:
        self._p = {p.profile_id: p for p in profiles}

    def get(self, pid: str) -> Profile | None:
        return self._p.get(pid)

    def get_all_sample(self) -> dict[str, Profile]:
        return dict(self._p)

    def get_ingested(self) -> dict[str, Profile]:
        return {}


def _profile(pid: str) -> Profile:
    return Profile(
        profile_id=pid,
        source=ProfileSource.MANUAL,
        personal=PersonalInfo(name=f"User {pid}"),
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    # No `with` block: the lifespan (which loads FAISS/models) is not run.
    saved_orch = search_route._orchestrator
    saved_store = profiles_route._profile_store
    saved_ingest = ingest_route._ingest_service
    search_route._orchestrator = None
    profiles_route._profile_store = None
    ingest_route._ingest_service = None
    init_health(index_size=0)
    try:
        yield TestClient(main_module.app, raise_server_exceptions=False)
    finally:
        search_route._orchestrator = saved_orch
        profiles_route._profile_store = saved_store
        ingest_route._ingest_service = saved_ingest
        init_health(index_size=0)


# ── health ──────────────────────────────────────────────────────────


def test_health_degraded_when_index_empty(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["index_size"] == 0
    assert body["models_loaded"] == {"embedding": False, "cross_encoder": False}


def test_health_healthy_after_index_and_models(client: TestClient) -> None:
    set_index_size(123)
    set_model_loaded("embedding")
    r = client.get("/api/v1/health")
    body = r.json()
    assert body["status"] == "healthy"
    assert body["index_size"] == 123
    assert body["models_loaded"]["embedding"] is True
    assert body["models_loaded"]["cross_encoder"] is False


# ── search ──────────────────────────────────────────────────────────


def test_search_503_when_not_initialized(client: TestClient) -> None:
    r = client.post("/api/v1/search", json={"query": "python developer"})
    assert r.status_code == 503
    assert "not initialized" in r.json()["detail"]


def test_search_happy_path_forwards_request_fields(client: TestClient) -> None:
    fake = FakeOrchestrator()
    search_route.init_orchestrator(fake)  # type: ignore[arg-type]
    r = client.post(
        "/api/v1/search",
        json={
            "query": "ML engineer in Bangalore",
            "max_results": 5,
            "use_turbo": True,
            "slider_weights": {"skill_match": 70},
            "filters": {"location": "Bangalore", "remote_ok": True},
        },
    )
    assert r.status_code == 200
    assert r.json()["query_id"] == "q-1"
    assert r.json()["total_candidates_searched"] == 42
    (call,) = fake.calls
    assert call["raw_query"] == "ML engineer in Bangalore"
    assert call["top_k"] == 5
    assert call["use_turbo"] is True
    assert call["slider_weights"] == {"skill_match": 70}
    assert isinstance(call["filters"], SearchFilters)
    assert call["filters"].location == "Bangalore"
    assert call["filters"].remote_ok is True


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "x" * 2001},
        {"query": "python", "max_results": 0},
        {"query": "python", "max_results": 101},
        {"query": "python", "max_results": "lots"},
        {"query": "python", "slider_weights": {"skills": "high"}},
    ],
)
def test_search_validation_errors_are_422(client: TestClient, payload: dict[str, Any]) -> None:
    fake = FakeOrchestrator()
    search_route.init_orchestrator(fake)  # type: ignore[arg-type]
    r = client.post("/api/v1/search", json=payload)
    assert r.status_code == 422
    assert fake.calls == []


def test_search_malformed_json_body_is_422(client: TestClient) -> None:
    search_route.init_orchestrator(FakeOrchestrator())  # type: ignore[arg-type]
    r = client.post(
        "/api/v1/search", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert r.status_code == 422


def test_search_failure_is_500_without_leaking_internals(client: TestClient) -> None:
    # Regression: the 500 detail used to echo the raw exception text
    # (file paths, provider errors) back to the client.
    search_route.init_orchestrator(FakeOrchestrator(fail=True))  # type: ignore[arg-type]
    r = client.post("/api/v1/search", json={"query": "python"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Search failed"}
    assert "C:/models" not in r.text


def test_search_wrong_method_is_405(client: TestClient) -> None:
    assert client.get("/api/v1/search").status_code == 405


# ── profiles ────────────────────────────────────────────────────────


def test_profiles_503_when_not_initialized(client: TestClient) -> None:
    assert client.get("/api/v1/profiles/abc").status_code == 503
    assert client.get("/api/v1/profiles").status_code == 503


def test_get_profile_found_and_missing(client: TestClient) -> None:
    profiles_route.init_profiles(FakeProfileStore([_profile("p1")]))  # type: ignore[arg-type]
    r = client.get("/api/v1/profiles/p1")
    assert r.status_code == 200
    assert r.json()["profile_id"] == "p1"
    assert r.json()["personal"]["name"] == "User p1"

    r = client.get("/api/v1/profiles/nope")
    assert r.status_code == 404
    assert "nope" in r.json()["detail"]


def test_list_profiles_pagination(client: TestClient) -> None:
    store = FakeProfileStore([_profile(f"p{i}") for i in range(30)])
    profiles_route.init_profiles(store)  # type: ignore[arg-type]
    r = client.get("/api/v1/profiles")
    assert r.status_code == 200
    assert len(r.json()) == 20  # default limit
    r = client.get("/api/v1/profiles", params={"skip": 25, "limit": 10})
    assert [p["profile_id"] for p in r.json()] == ["p25", "p26", "p27", "p28", "p29"]
    r = client.get("/api/v1/profiles", params={"skip": 100})
    assert r.json() == []


@pytest.mark.parametrize(
    "params",
    [{"skip": -5}, {"limit": 0}, {"limit": -1}, {"limit": 101}, {"skip": "abc"}],
)
def test_list_profiles_rejects_bad_pagination(
    client: TestClient, params: dict[str, Any]
) -> None:
    # Regression: negative skip/limit were passed straight into a slice, so
    # skip=-5 returned the *last* 5 profiles and limit=-1 dropped one silently.
    store = FakeProfileStore([_profile(f"p{i}") for i in range(30)])
    profiles_route.init_profiles(store)  # type: ignore[arg-type]
    assert client.get("/api/v1/profiles", params=params).status_code == 422


# ── ingest ──────────────────────────────────────────────────────────


def _upload(client: TestClient, name: str, content: bytes) -> Any:
    return client.post(
        "/api/v1/ingest", files={"file": (name, content, "application/json")}
    )


def test_ingest_without_index_is_503(client: TestClient) -> None:
    # Valid upload, but no index/ingest service (lifespan found no FAISS index).
    r = _upload(client, "p.json", json.dumps([{"candidate_id": "A"}]).encode())
    assert r.status_code == 503
    assert "build_indexes" in r.json()["detail"]


def test_ingest_rejects_non_json_extension(client: TestClient) -> None:
    r = _upload(client, "p.csv", b"a,b")
    assert r.status_code == 400
    assert ".json" in r.json()["detail"]


def test_ingest_missing_file_is_422(client: TestClient) -> None:
    assert client.post("/api/v1/ingest").status_code == 422


def test_ingest_invalid_json_is_400(client: TestClient) -> None:
    r = _upload(client, "p.json", b"{oops")
    assert r.status_code == 400
    assert r.json()["detail"].startswith("Invalid JSON")


def test_ingest_non_utf8_is_400(client: TestClient) -> None:
    r = _upload(client, "p.json", b"\xff\xfe\x00garbage")
    assert r.status_code == 400


@pytest.mark.parametrize("body", [b"5", b'"text"', b"null", b"true"])
def test_ingest_scalar_json_is_400(client: TestClient, body: bytes) -> None:
    # Regression: a JSON scalar crashed with TypeError (len(5)) -> 500.
    r = _upload(client, "p.json", body)
    assert r.status_code == 400


# ── middleware ──────────────────────────────────────────────────────


def test_body_too_large_is_413(client: TestClient) -> None:
    r = client.post(
        "/api/v1/search",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": str(11 * 1024 * 1024)},
    )
    assert r.status_code == 413
    assert "too large" in r.json()["detail"]


async def _call_asgi(headers: list[tuple[bytes, bytes]]) -> tuple[int, bytes]:
    """Drive the real app over raw ASGI so we can send headers httpx won't."""
    scope: dict[str, Any] = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": "/api/v1/health", "raw_path": b"",
        "query_string": b"", "root_path": "", "headers": headers,
        "client": ("test", 1), "server": ("test", 80),
    }
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await main_module.app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], body


@pytest.mark.parametrize("value", [b"abc", b"12x", b"-1"])
async def test_malformed_content_length_is_400(value: bytes) -> None:
    # Regression: int("abc") raised inside the middleware -> 500.
    status, body = await _call_asgi([(b"content-length", value)])
    assert status == 400
    assert b"Content-Length" in body


async def test_valid_content_length_passes_through() -> None:
    status, _ = await _call_asgi([(b"content-length", b"0")])
    assert status == 200


def test_request_logging_middleware_logs_method_path_status(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="api.access"):
        client.get("/api/v1/health")
        client.get("/api/v1/does-not-exist")
    lines = [r.getMessage() for r in caplog.records if r.name == "api.access"]
    assert any(m.startswith("GET /api/v1/health status=200 duration=") for m in lines)
    assert any("/api/v1/does-not-exist status=404" in m for m in lines)


def test_unknown_route_is_404(client: TestClient) -> None:
    assert client.get("/api/v1/nope").status_code == 404


# ── lifespan ────────────────────────────────────────────────────────


def test_lifespan_without_index_starts_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_module, "DATA_DIR", tmp_path)
    monkeypatch.setenv("AUTO_INDEX", "false")
    with TestClient(main_module.app) as c:
        assert (tmp_path / "indexes").is_dir()
        r = c.get("/api/v1/health")
        assert r.json()["status"] == "degraded"
        # Search stays unavailable rather than crashing.
        saved = search_route._orchestrator
        search_route._orchestrator = None
        try:
            assert c.post("/api/v1/search", json={"query": "x"}).status_code == 503
        finally:
            search_route._orchestrator = saved


def test_lifespan_with_index_wires_orchestrator_and_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "indexes").mkdir()
    (tmp_path / "indexes" / "faiss_index.bin").write_bytes(b"")
    monkeypatch.setattr(main_module, "DATA_DIR", tmp_path)

    class FakeVectorSearch:
        size = 7

    store = FakeProfileStore([_profile("p1")])
    orch = FakeOrchestrator()
    # The lifespan wires IngestService from the executor's search components.
    hybrid = type("H", (), {"bm25_search": object(), "embedder": object(), "_cache": None})()
    orch.executor = type("E", (), {"hybrid_search": hybrid})()  # type: ignore[attr-defined]
    monkeypatch.setenv("INGESTED_PROFILES_PATH", str(tmp_path / "ingested.jsonl"))
    seen: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> tuple[Any, Any, Any]:
        seen.update(kwargs)
        return orch, FakeVectorSearch(), store

    import src.core.config as config_module

    monkeypatch.setattr(config_module, "build_orchestrator", fake_build)
    saved_orch, saved_store = search_route._orchestrator, profiles_route._profile_store
    saved_ingest = ingest_route._ingest_service
    try:
        with TestClient(main_module.app) as c:
            assert ingest_route._ingest_service is not None
            health = c.get("/api/v1/health").json()
            assert health["status"] == "healthy"
            assert health["index_size"] == 7
            assert health["models_loaded"] == {"embedding": True, "cross_encoder": True}
            assert c.post("/api/v1/search", json={"query": "x"}).status_code == 200
            assert c.get("/api/v1/profiles/p1").status_code == 200
        assert seen["faiss_path"] == tmp_path / "indexes" / "faiss_index.bin"
        assert seen["bm25_path"] == tmp_path / "indexes" / "bm25_index.pkl"
    finally:
        search_route._orchestrator, profiles_route._profile_store = saved_orch, saved_store
        ingest_route._ingest_service = saved_ingest
        init_health(index_size=0)
