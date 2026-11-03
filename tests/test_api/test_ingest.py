"""POST /api/v1/ingest end to end: upload -> persist -> index -> /search finds it.

Runs the real FastAPI lifespan, Orchestrator (turbo path), ExecutorAgent,
HybridSearch, FAISS, BM25 and ProfileStore. Only the models are faked: a
hashed bag-of-words embedder and a word-overlap reranker (no downloads, no
LLM), so the suite stays offline and small.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import src.api.routes.ingest as ingest_route
import src.api.routes.profiles as profiles_route
import src.api.routes.search as search_route
import src.core.config as config_module
import src.main as main_module
from src.api.routes.health import init_health
from src.core.profile_store import ProfileStore
from src.ingestion.service import MAX_PROFILES_PER_REQUEST
from tests.test_ingestion.test_service import Stack, raw_profile


class OverlapReranker:
    def rerank(
        self, query: str, candidates: list[tuple[str, str, float]], top_k: int = 10
    ) -> list[tuple[str, float]]:
        q = set(query.lower().split())
        scored = [
            (pid, len(q & set(text.lower().split())) / (len(q) or 1) + 0.01 * s)
            for pid, text, s in candidates
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]


@pytest.fixture
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Lifespan with a real (empty) search stack; yields a factory for clients."""
    import src.core.profile_store as ps_module

    data = tmp_path / "data"
    (data / "indexes").mkdir(parents=True)
    (data / "indexes" / "faiss_index.bin").write_bytes(b"")  # lifespan only checks existence
    monkeypatch.setattr(main_module, "DATA_DIR", data)
    monkeypatch.setattr(ps_module, "DATA_DIR", data)
    monkeypatch.setattr(ps_module, "SAMPLE_PATH", data / "none.json")
    monkeypatch.delenv("PROFILES_PATH", raising=False)
    ingested = data / "profiles" / "ingested.jsonl"
    monkeypatch.setenv("INGESTED_PROFILES_PATH", str(ingested))

    stacks: list[Stack] = []

    def fake_build(**_: Any) -> tuple[Any, Any, Any]:
        from src.agents.executor import ExecutorAgent
        from src.agents.orchestrator import Orchestrator
        from src.matching.scorer import CandidateScorer
        from src.search.hybrid import HybridSearch

        stack = Stack(ProfileStore(jsonl_path=data / "missing.jsonl"), ingested)
        stacks.append(stack)
        hybrid = HybridSearch(stack.vector, stack.bm25, stack.embedder)  # type: ignore[arg-type]
        executor = ExecutorAgent(
            hybrid, OverlapReranker(), CandidateScorer(), stack.store  # type: ignore[arg-type]
        )
        orch = Orchestrator(MagicMock(), executor, MagicMock())
        return orch, stack.vector, stack.store

    monkeypatch.setattr(config_module, "build_orchestrator", fake_build)
    saved = (
        search_route._orchestrator, profiles_route._profile_store, ingest_route._ingest_service
    )
    try:
        yield {"stacks": stacks, "ingested": ingested}
    finally:
        (search_route._orchestrator, profiles_route._profile_store,
         ingest_route._ingest_service) = saved
        init_health(index_size=0)


def _upload(c: TestClient, payload: Any, name: str = "profiles.json") -> Any:
    body = json.dumps(payload).encode()
    return c.post("/api/v1/ingest", files={"file": (name, body, "application/json")})


def _search(c: TestClient, query: str) -> list[str]:
    r = c.post("/api/v1/search", json={"query": query, "use_turbo": True, "max_results": 5})
    assert r.status_code == 200, r.text
    return [item["profile_id"] for item in r.json()["results"]]


PROFILES = [
    raw_profile("K8S-1", "Kubernetes Platform Engineer", ("Kubernetes", "Terraform"),
                "Runs Kubernetes clusters on AWS with Terraform."),
    raw_profile("CHEF-1", "Pastry Chef", ("Baking",), "Bakes croissants and bread."),
    raw_profile("ACC-1", "Accountant", ("Excel", "Tally"), "Prepares GST filings."),
]


def test_ingest_then_search_and_fetch(app_env: dict[str, Any]) -> None:
    with TestClient(main_module.app) as c:
        assert _search(c, "Kubernetes Terraform engineer") == []  # empty index

        r = _upload(c, PROFILES)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total_profiles"] == 3
        assert (body["successful"], body["failed"], body["created"]) == (3, 0, 3)
        assert body["profile_ids"] == ["K8S-1", "CHEF-1", "ACC-1"]

        assert _search(c, "Kubernetes Terraform engineer")[0] == "K8S-1"
        assert _search(c, "pastry chef baking")[0] == "CHEF-1"

        p = c.get("/api/v1/profiles/ACC-1")
        assert p.status_code == 200
        assert p.json()["professional"]["current_title"] == "Accountant"
        listed = [x["profile_id"] for x in c.get("/api/v1/profiles").json()]
        assert set(listed) == {"K8S-1", "CHEF-1", "ACC-1"}
        assert c.get("/api/v1/health").json()["index_size"] == 3

    persisted = app_env["ingested"].read_text(encoding="utf-8").splitlines()
    assert len(persisted) == 3


def test_reingest_is_idempotent_and_updates_search(app_env: dict[str, Any]) -> None:
    with TestClient(main_module.app) as c:
        _upload(c, PROFILES)
        _search(c, "pastry chef baking")  # warm any result cache

        again = _upload(c, PROFILES).json()
        assert (again["created"], again["updated"], again["unchanged"]) == (0, 0, 3)
        assert again["successful"] == 3
        assert c.get("/api/v1/health").json()["index_size"] == 3

        changed = raw_profile("CHEF-1", "Rust Developer", ("Rust",), "Writes Rust compilers.")
        r = _upload(c, changed, name="one.json").json()  # single object form
        assert (r["created"], r["updated"], r["unchanged"]) == (0, 1, 0)
        assert c.get("/api/v1/health").json()["index_size"] == 3
        assert _search(c, "Rust compilers developer")[0] == "CHEF-1"
        title = c.get("/api/v1/profiles/CHEF-1").json()["professional"]["current_title"]
        assert title == "Rust Developer"

    assert len(app_env["ingested"].read_text(encoding="utf-8").splitlines()) == 3


def test_ingested_profiles_survive_restart(app_env: dict[str, Any]) -> None:
    with TestClient(main_module.app) as c:
        _upload(c, PROFILES)
    with TestClient(main_module.app) as c:  # new lifespan, fresh empty indexes
        assert len(app_env["stacks"]) == 2
        assert c.get("/api/v1/health").json()["index_size"] == 3
        assert _search(c, "Kubernetes Terraform engineer")[0] == "K8S-1"
        assert _upload(c, PROFILES).json()["unchanged"] == 3


def test_partial_failures_are_reported_per_item(app_env: dict[str, Any]) -> None:
    with TestClient(main_module.app) as c:
        r = _upload(c, [PROFILES[0], 7, {"profile": {"summary": "no id"}},
                        {"candidate_id": "S", "skills": [{"proficiency": "x"}]}])
        assert r.status_code == 200
        body = r.json()
        assert (body["total_profiles"], body["successful"], body["failed"]) == (4, 1, 3)
        assert body["errors"][0].startswith("Item 1: expected a JSON object")
        assert "missing profile id" in body["errors"][1]
        assert "skills[0]" in body["errors"][2]
        assert body["profile_ids"] == ["K8S-1"]


def test_too_many_profiles_is_400(app_env: dict[str, Any]) -> None:
    with TestClient(main_module.app) as c:
        r = _upload(c, [{"candidate_id": str(i)} for i in range(MAX_PROFILES_PER_REQUEST + 1)])
        assert r.status_code == 400
        assert "Too many profiles" in r.json()["detail"]
        assert not app_env["ingested"].exists()


def test_index_failure_is_500_without_leaking(
    app_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    with TestClient(main_module.app, raise_server_exceptions=False) as c:
        stack = app_env["stacks"][0]

        def boom(*_: Any, **__: Any) -> Any:
            raise RuntimeError("C:/secret/model/path exploded")

        monkeypatch.setattr(stack.embedder, "embed_batch", boom)
        r = _upload(c, PROFILES[:1])
        assert r.status_code == 500
        assert r.json()["detail"] == "Ingest failed"
        # Persisted before indexing, so a restart replays it.
    assert app_env["ingested"].exists()
