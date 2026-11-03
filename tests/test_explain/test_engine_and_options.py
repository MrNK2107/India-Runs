from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.main as main_module
from src.core.models import Profile, WorkExperience
from src.language.hashing import HashingEmbedder
from src.matching.behavioral_scorer import compute_career_trajectory

SAMPLE_100 = Path(__file__).resolve().parents[2] / "data" / "samples" / "sample_candidates.json"


def test_hashing_embedder_is_deterministic_normalised_and_lexical() -> None:
    e = HashingEmbedder()
    a = e.embed("Senior Spark data engineer in Bengaluru")
    assert a.shape == (384,) and a.dtype == np.float32
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    assert np.array_equal(a, HashingEmbedder().embed("Senior Spark data engineer in Bengaluru"))
    near = e.embed("spark data engineer bangalore")
    far = e.embed("pastry chef croissants")
    assert float(a @ near) > float(a @ far) + 0.2
    assert e.embed_batch([]).shape == (0, 384)
    assert e.embed("").sum() == 0


def test_create_embedder_backend_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.language.multilingual import MultilingualEmbedder, create_embedder

    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    assert isinstance(create_embedder(), HashingEmbedder)
    monkeypatch.setenv("EMBEDDING_BACKEND", "transformer")
    assert isinstance(create_embedder(), MultilingualEmbedder)
    monkeypatch.setenv("EMBEDDING_BACKEND", "nope")
    with pytest.raises(ValueError):
        create_embedder()


def _hopper(current_start: str) -> Profile:
    return Profile(
        personal={"name": "x"},  # type: ignore[arg-type]
        experience=[
            WorkExperience(title="Engineer", company="A", start_date=current_start,
                           end_date=None, is_current=True),
            WorkExperience(title="Engineer", company="B", start_date="2021-01", end_date="2022-01"),
            WorkExperience(title="Engineer", company="C", start_date="2020-01", end_date="2021-01"),
        ],
    )


def test_trajectory_default_unchanged_and_include_current_is_opt_in() -> None:
    p = _hopper("2018-01")  # 8 years in the current job
    default = compute_career_trajectory(p)
    assert default == compute_career_trajectory(p, include_current_tenure=False)
    with_current = compute_career_trajectory(p, include_current_tenure=True, as_of="2026-01")
    # completed roles avg 12 months -> job hopper (0.3); with the 96-month current
    # role the average is 40 months -> strong stability (0.9)
    assert with_current > default


@pytest.fixture
def sample_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(main_module, "DATA_DIR", tmp_path)
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("CROSS_ENCODER_ENABLED", "false")
    monkeypatch.setenv("SAMPLE_PROFILES_PATH", str(SAMPLE_100))
    monkeypatch.setenv("INGESTED_PROFILES_PATH", str(tmp_path / "ingested.jsonl"))
    from src.core.config import get_settings
    get_settings.cache_clear()
    with TestClient(main_module.app) as c:
        yield c
    get_settings.cache_clear()


def test_zero_setup_sample_engine_serves_explained_results(sample_app: TestClient) -> None:
    health = sample_app.get("/api/v1/health").json()
    assert health["status"] == "healthy"
    assert health["engine"] == {"mode": "sample", "dataset": "sample_candidates.json",
                                "embedding_backend": "hashing"}
    assert health["index_size"] == 100

    r = sample_app.post("/api/v1/search", json={
        "query": "data engineer with spark and airflow", "use_turbo": True, "max_results": 5,
    })
    assert r.status_code == 200
    results = r.json()["results"]
    assert results
    ex = results[0]["explanation"]
    assert ex["profile_id"] == results[0]["profile_id"]
    assert abs(sum(c["points"] for c in ex["contributions"]) - ex["overall"]) < 2e-3
    assert {q["kind"] for q in ex["requirements"]} >= {"skill"}

    no_ex = sample_app.post("/api/v1/search", json={
        "query": "data engineer", "use_turbo": True, "max_results": 2, "explain": False,
    }).json()
    assert all(item["explanation"] is None for item in no_ex["results"])


def test_explain_endpoint_and_trajectory_mode(sample_app: TestClient) -> None:
    r = sample_app.post("/api/v1/explain", json={
        "query": "backend engineer with python and kafka", "profile_id": "CAND_0000001",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["profile_id"] == "CAND_0000001"
    assert body["trajectory"]["mode"] == "completed_roles"
    r2 = sample_app.post("/api/v1/explain", json={
        "query": "backend engineer with python and kafka", "profile_id": "CAND_0000001",
        "trajectory_mode": "include_current",
    })
    assert r2.json()["trajectory"]["mode"] == "include_current"
    assert sample_app.post("/api/v1/explain", json={
        "query": "x", "profile_id": "NOPE"}).status_code == 404
    assert sample_app.post("/api/v1/search", json={
        "query": "x", "trajectory_mode": "sideways"}).status_code == 422


def test_ingested_profile_is_explained_with_its_own_evidence(sample_app: TestClient) -> None:
    files = {"file": ("p.json", (Path(__file__).resolve().parents[2] / "data" / "samples"
                                 / "ingest_example.json").read_bytes(), "application/json")}
    assert sample_app.post("/api/v1/ingest", files=files).json()["created"] == 1
    r = sample_app.post("/api/v1/search", json={
        "query": "Flutter developer with Dart and Firebase", "use_turbo": True, "max_results": 3,
    }).json()
    top = r["results"][0]
    assert top["profile_id"] == "DEMO_0001"
    reqs = {q["requirement"].lower(): q for q in top["explanation"]["requirements"]}
    assert reqs["flutter"]["status"] == "met"
    assert reqs["flutter"]["evidence"][0]["source"] in {"skills", "assessment"}


def test_skill_evidence_work_history_is_opt_in(sample_app: TestClient) -> None:
    q = {"query": "data engineer with spark and airflow", "use_turbo": True, "max_results": 10}
    listed = sample_app.post("/api/v1/search", json=q).json()["results"]
    credited = sample_app.post("/api/v1/search",
                               json={**q, "skill_evidence": "work_history"}).json()["results"]
    by_id = {r["profile_id"]: r for r in listed}
    # Default ranking is unchanged by the option's existence; opting in can only add
    # matched skills (never remove) for the same candidate.
    gained = 0
    for r in credited:
        if r["profile_id"] in by_id:
            before = set(by_id[r["profile_id"]]["matched_skills"])
            assert before <= set(r["matched_skills"])
            gained += len(set(r["matched_skills"]) - before)
    assert gained > 0
    assert any("Credited from work history" in s
               for r in credited for s in r["explanation"]["strengths"])
    bad = sample_app.post("/api/v1/search", json={**q, "skill_evidence": "vibes"})
    assert bad.status_code == 422


def test_credit_work_history_ignores_summary_only_claims() -> None:
    from src.agents.executor import _credit_work_history
    from src.core.models import PersonalInfo

    p = Profile(
        personal=PersonalInfo(name="x"),
        raw_text="Summary: I know Rust and Spark.",
        experience=[WorkExperience(title="Engineer", company="A",
                                   description="Ran Spark jobs nightly.")],
    )
    matched, missing = _credit_work_history([], ["spark", "rust"], p)
    assert matched == ["spark"] and missing == ["rust"]


def test_web_app_is_served(sample_app: TestClient) -> None:
    page = sample_app.get("/")
    assert page.status_code == 200 and "Kasauti" in page.text
    assert 'src="/static/app.js"' in page.text
    for asset in ("app.js", "app.css", "mark.svg", "example-profile.json"):
        assert sample_app.get(f"/static/{asset}").status_code == 200
    js = sample_app.get("/static/app.js").text
    # untrusted profile data must be escaped before it touches innerHTML
    assert "function esc(" in js and "&lt;" in js


def test_verdict_matches_displayed_score() -> None:
    from src.core.models import PersonalInfo, Profile
    from src.core.query_parser import parse_query
    from src.explain.evidence import explain_candidate

    p = Profile(personal=PersonalInfo(name="x"))
    ex = explain_candidate(p, parse_query("python"), {"skill_match": 0.6956}, 0.6956,
                           {"skill_match": 1.0})
    assert ex.verdict == "strong"  # shown as 70/100


def test_fairness_report_for_a_result_list(sample_app: TestClient) -> None:
    results = sample_app.post("/api/v1/search", json={
        "query": "software engineer", "use_turbo": True, "max_results": 20,
    }).json()["results"]
    items = [{"profile_id": r["profile_id"], "score": r["scores"]["overall"]} for r in results]
    rep = sample_app.post("/api/v1/fairness", json={"items": items}).json()
    assert rep["candidates"] == len(items)
    assert [d["dimension"] for d in rep["dimensions"]] == ["location", "university", "language"]
    assert all(d["status"] in {"ok", "flag", "insufficient"} for d in rep["dimensions"])
    assert rep["flagged"] == sum(d["status"] == "flag" for d in rep["dimensions"])
    # all-English sample -> nothing to compare against
    assert rep["dimensions"][2]["status"] == "insufficient"
    assert sample_app.post("/api/v1/fairness", json={"items": []}).status_code == 422
    unknown = sample_app.post("/api/v1/fairness",
                              json={"items": [{"profile_id": "NOPE", "score": 0.5}]})
    assert unknown.json()["candidates"] == 0


def test_points_sum_to_overall_for_every_result(sample_app: TestClient) -> None:
    # includes candidates found by only one retrieval channel (no vector or no BM25 score)
    for q in ("software engineer with python", "data engineer with spark and airflow",
              "frontend developer with react in pune"):
        for r in sample_app.post("/api/v1/search", json={
                "query": q, "use_turbo": True, "max_results": 25}).json()["results"]:
            ex = r["explanation"]
            assert abs(sum(c["points"] for c in ex["contributions"]) - ex["overall"]) < 2e-3, (q, r)


def test_failed_sample_engine_starts_degraded(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "DATA_DIR", tmp_path)
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("SAMPLE_PROFILES_PATH", str(tmp_path / "missing.jsonl"))
    with TestClient(main_module.app) as c:
        assert c.get("/api/v1/health").json()["status"] == "degraded"


def test_explain_rejects_blank_query(sample_app: TestClient) -> None:
    r = sample_app.post("/api/v1/explain", json={"query": "   ", "profile_id": "CAND_0000001"})
    assert r.status_code == 422
