"""IngestService: validation, persistence, indexing, idempotency, replay.

Real FAISS / BM25 / ProfileStore / HybridSearch / ExecutorAgent; only the
sentence-transformer is replaced by a deterministic hashed bag-of-words
embedder, so nothing is downloaded and memory stays small.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import src.core.profile_store as ps_module
from src.core.models import ParsedQuery, RequiredSkill
from src.core.profile_store import ProfileStore
from src.ingestion.service import (
    IngestService,
    ProfileValidationError,
    build_document_text,
    validate_and_normalize,
)
from src.search.bm25_search import BM25Search
from src.search.vector_search import VectorSearch

DIM = 64


class FakeEmbedder:
    """Hashed bag-of-words, L2-normalised: similar texts -> similar vectors."""

    dimension = DIM

    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray[Any, Any]:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray[Any, Any]:
        self.calls += 1
        return np.stack([self._vec(t) for t in texts])

    def embed(self, text: str) -> np.ndarray[Any, Any]:
        return self._vec(text)

    def embed_query(self, text: str) -> np.ndarray[Any, Any]:
        return self._vec(text)


def raw_profile(pid: str, title: str = "Backend Engineer", skills: tuple[str, ...] = ("Python",),
                summary: str = "Builds backend services.", **extra: Any) -> dict[str, Any]:
    return {
        "candidate_id": pid,
        "profile": {
            "anonymized_name": f"Person {pid}",
            "headline": title,
            "summary": summary,
            "location": "Bengaluru, Karnataka",
            "country": "India",
            "years_of_experience": 5,
            "current_title": title,
            "current_company": "Acme",
        },
        "career_history": [
            {"company": "Acme", "title": title, "start_date": "2021-01-01",
             "end_date": None, "is_current": True, "description": summary},
        ],
        "education": [],
        "skills": [{"name": s, "proficiency": "advanced"} for s in skills],
        **extra,
    }


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProfileStore:
    monkeypatch.delenv("PROFILES_PATH", raising=False)
    monkeypatch.setattr(ps_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ps_module, "SAMPLE_PATH", tmp_path / "none.json")
    return ProfileStore(jsonl_path=tmp_path / "missing.jsonl")


class Stack:
    def __init__(self, store: ProfileStore, path: Path) -> None:
        self.store = store
        self.vector = VectorSearch(dimension=DIM)
        self.bm25 = BM25Search()
        self.embedder = FakeEmbedder()
        self.changes = 0
        self.path = path
        self.service = IngestService(
            store, self.vector, self.bm25, self.embedder, path, on_change=self._changed
        )

    def _changed(self) -> None:
        self.changes += 1


@pytest.fixture
def stack(store: ProfileStore, tmp_path: Path) -> Stack:
    return Stack(store, tmp_path / "profiles" / "ingested.jsonl")


# ── validation ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "msg"),
    [
        (7, "expected a JSON object"),
        ({"profile": {"summary": "x"}}, "missing profile id"),
        ({"candidate_id": "  "}, "missing profile id"),
        ({"candidate_id": True}, "missing profile id"),
        ({"candidate_id": "x" * 200}, "longer than"),
        ({"candidate_id": "A", "skills": "Python"}, "'skills' must be a list"),
        ({"candidate_id": "A", "profile": "text"}, "'profile' must be a dict"),
        ({"candidate_id": "A", "skills": [{"proficiency": "x"}]}, "skills[0]"),
        ({"candidate_id": "A", "career_history": [{"title": "Dev"}]}, "company"),
        ({"candidate_id": "A", "education": [{}]}, "institution"),
        ({"candidate_id": "A"}, "no searchable content"),
    ],
)
def test_validation_errors(raw: Any, msg: str) -> None:
    with pytest.raises(ProfileValidationError, match=msg.replace("[", r"\[").replace("]", r"\]")):
        validate_and_normalize(raw)


@pytest.mark.parametrize("key", ["candidate_id", "profile_id", "id"])
def test_id_aliases_give_stable_ids(key: str) -> None:
    raw = raw_profile("X")
    del raw["candidate_id"]
    raw[key] = 42
    pid, profile = validate_and_normalize(raw)
    assert pid == profile.profile_id == "42"


def test_null_optional_sections_are_accepted() -> None:
    raw = raw_profile("N", education=None)
    raw["education"] = None
    pid, profile = validate_and_normalize(raw)
    assert pid == "N"
    assert profile.education == []


# ── ingest + retrieval ──────────────────────────────────────────────


def test_ingest_persists_indexes_and_stores(stack: Stack) -> None:
    res = stack.service.ingest([
        raw_profile("P1", "Kotlin Android Developer", ("Kotlin", "Android")),
        raw_profile("P2", "Data Engineer", ("Spark", "Airflow"), "Builds Spark pipelines."),
        raw_profile("P3", "Accountant", ("Excel",), "Prepares tax filings."),
    ])
    assert res.created == ["P1", "P2", "P3"]
    assert res.errors == []
    assert stack.changes == 1

    # persisted
    lines = [json.loads(x) for x in stack.path.read_text(encoding="utf-8").splitlines()]
    assert [x["profile_id"] for x in lines] == ["P1", "P2", "P3"]
    assert lines[0]["raw"]["candidate_id"] == "P1"

    # store
    p1 = stack.store.get("P1")
    assert p1 is not None and p1.professional is not None
    assert p1.professional.current_title == "Kotlin Android Developer"
    assert "P2" in stack.store and "P2" in stack.store.get_all_pids()

    # both indexes
    assert stack.vector.size == 3
    assert stack.bm25.size == 3
    assert stack.bm25.search("spark airflow", top_k=1)[0][0] == "P2"
    q = stack.embedder.embed_query(build_document_text(p1))
    assert stack.vector.search(q, top_k=1)[0][0] == "P1"


def test_reingest_identical_is_noop(stack: Stack) -> None:
    raw = raw_profile("P1")
    stack.service.ingest([raw])
    calls, mtime = stack.embedder.calls, stack.path.stat().st_mtime_ns
    res = stack.service.ingest([json.loads(json.dumps(raw))])
    assert res.unchanged == ["P1"] and res.created == [] and res.updated == []
    assert stack.embedder.calls == calls  # no re-embedding
    assert stack.path.stat().st_mtime_ns == mtime  # no rewrite
    assert stack.vector.size == 1 and stack.bm25.size == 1
    assert stack.changes == 1


def test_reingest_changed_replaces_everywhere(stack: Stack) -> None:
    stack.service.ingest([raw_profile("P1", "Java Developer", ("Java",), "Java Spring services.")])
    stack.service.ingest([raw_profile("P9", "Designer", ("Figma",), "Figma mockups.")])
    res = stack.service.ingest(
        [raw_profile("P1", "Rust Developer", ("Rust",), "Rust systems programming.")]
    )
    assert res.updated == ["P1"]

    p = stack.store.get("P1")
    assert p is not None and p.professional is not None
    assert p.professional.current_title == "Rust Developer"
    assert stack.vector.size == 2  # old P1 vector tombstoned, not counted
    assert stack.bm25.size == 2
    p1_tokens = stack.bm25.corpus_tokenized[stack.bm25.id_map.index("P1")]
    assert "rust" in p1_tokens and "java" not in p1_tokens
    hits = stack.vector.search(stack.embedder.embed_query("Rust systems programming"), top_k=5)
    assert [pid for pid, _ in hits].count("P1") == 1
    assert len(stack.path.read_text(encoding="utf-8").splitlines()) == 2


def test_duplicate_ids_in_one_upload_last_wins(stack: Stack) -> None:
    res = stack.service.ingest([raw_profile("D", "First"), raw_profile("D", "Second")])
    assert res.created == ["D"]
    p = stack.store.get("D")
    assert p is not None and p.professional is not None
    assert p.professional.current_title == "Second"
    assert stack.vector.size == 1


def test_partial_failure_keeps_good_items(stack: Stack) -> None:
    res = stack.service.ingest([raw_profile("G"), {"profile": {}}, 3])
    assert res.created == ["G"]
    assert len(res.errors) == 2
    assert res.errors[0].startswith("Item 1:")
    assert res.errors[1].startswith("Item 2:")


def test_all_invalid_writes_nothing(stack: Stack) -> None:
    res = stack.service.ingest([{"nope": 1}])
    assert res.successful == 0
    assert not stack.path.exists()
    assert stack.changes == 0


def test_replay_on_restart(stack: Stack, store: ProfileStore, tmp_path: Path) -> None:
    stack.service.ingest([raw_profile("R1", skills=("Go",)), raw_profile("R2", skills=("Elixir",))])
    stack.service.ingest([raw_profile("R1", "Go Developer", ("Go",))])

    fresh_store = ProfileStore(jsonl_path=tmp_path / "missing.jsonl")
    fresh = Stack(fresh_store, stack.path)
    assert fresh.service.load() == 2
    assert len(fresh.service) == 2
    p = fresh_store.get("R1")
    assert p is not None and p.professional is not None
    assert p.professional.current_title == "Go Developer"
    assert fresh.vector.size == 2 and fresh.bm25.size == 2
    # replayed profile is still idempotent after restart
    assert fresh.service.ingest([raw_profile("R2", skills=("Elixir",))]).unchanged == ["R2"]


def test_load_skips_corrupt_lines(stack: Stack) -> None:
    stack.path.parent.mkdir(parents=True)
    good = {"profile_id": "OK", "raw": raw_profile("OK")}
    stack.path.write_text(
        "not json\n" + json.dumps({"profile_id": "X", "raw": 5}) + "\n\n" + json.dumps(good) + "\n",
        encoding="utf-8",
    )
    assert stack.service.load() == 1
    assert stack.store.get("OK") is not None


def test_ingest_into_existing_base_index_replaces_base_vector(stack: Stack) -> None:
    # Prebuilt index (as from scripts/build_indexes.py) already holds B1.
    texts = ["old text about cobol mainframes", "unrelated marketing profile"]
    stack.vector.build_index(stack.embedder.embed_batch(texts), ["B1", "B2"])
    stack.bm25.build_index(texts, ["B1", "B2"])
    stack.service.ingest([raw_profile("B1", "Python Developer", ("Python",), "Python APIs.")])
    assert stack.vector.size == 2
    assert stack.bm25.size == 2
    assert stack.bm25.search("cobol mainframes", top_k=2)[0][1] == 0 or \
        stack.bm25.search("cobol mainframes", top_k=1)[0][0] != "B1"


@pytest.mark.asyncio
async def test_ingested_profile_reaches_executor_search_path(stack: Stack) -> None:
    """End to end through the real ExecutorAgent (hybrid retrieval + scoring)."""
    from src.agents.executor import ExecutorAgent
    from src.matching.scorer import CandidateScorer
    from src.search.hybrid import HybridSearch

    stack.service.ingest([
        raw_profile("E1", "Kubernetes Platform Engineer", ("Kubernetes", "Terraform"),
                    "Runs Kubernetes clusters with Terraform."),
        raw_profile("E2", "Pastry Chef", ("Baking",), "Bakes croissants."),
    ])
    hybrid = HybridSearch(stack.vector, stack.bm25, stack.embedder)  # type: ignore[arg-type]
    executor = ExecutorAgent(hybrid, None, CandidateScorer(), stack.store)  # type: ignore[arg-type]
    parsed = ParsedQuery(
        required_skills=[RequiredSkill(name="Kubernetes")],
        original_query="Kubernetes engineer",
    )
    results = await executor.execute(parsed, top_k=5, skip_reranker=True)
    assert results, "ingested profiles were not retrieved"
    assert results[0].profile_id == "E1"
