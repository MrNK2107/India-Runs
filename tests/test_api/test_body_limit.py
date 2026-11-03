"""The body-size limit must hold for chunked requests (no Content-Length), on every route."""
from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.api.routes.search as search_route
from src.api.middleware.validation import InputValidationMiddleware
from tests.test_api.test_routes import FakeOrchestrator


def _chunks(data: bytes, size: int = 64) -> Any:
    for i in range(0, len(data), size):
        yield data[i : i + size]


@pytest.fixture
def small_limit(monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(InputValidationMiddleware, "MAX_BODY_SIZE", 500)
    return 500


def test_chunked_search_body_over_limit_is_413(client: TestClient, small_limit: int) -> None:
    fake = FakeOrchestrator()
    search_route.init_orchestrator(fake)  # type: ignore[arg-type]
    body = json.dumps({"query": "python " * 100}).encode()
    assert len(body) > small_limit
    r = client.post(
        "/api/v1/search", content=_chunks(body),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 413
    assert "too large" in r.json()["detail"]
    assert fake.calls == []


def test_chunked_search_body_under_limit_passes(client: TestClient, small_limit: int) -> None:
    fake = FakeOrchestrator()
    search_route.init_orchestrator(fake)  # type: ignore[arg-type]
    body = json.dumps({"query": "python developer"}).encode()
    r = client.post(
        "/api/v1/search", content=_chunks(body, 5),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 200
    assert fake.calls[0]["raw_query"] == "python developer"


def test_chunked_ingest_over_limit_is_413(client: TestClient, small_limit: int) -> None:
    import src.api.routes.ingest as ingest_route

    ingest_route.init_ingest(None)
    boundary = "xB"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.json"\r\n'
        "Content-Type: application/json\r\n\r\n"
    )
    payload = b"[" + b"{}," * 400 + b"{}]"
    body = head.encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    r = client.post(
        "/api/v1/ingest", content=_chunks(body),
        headers={"content-type": f"multipart/form-data; boundary={boundary}"},
    )
    assert r.status_code == 413


def test_get_requests_unaffected(client: TestClient, small_limit: int) -> None:
    assert client.get("/api/v1/health").status_code == 200
