"""Unit tests for CrossEncoderReranker.

The real cross-encoder is replaced by ``FakeCrossEncoder`` (see tests/conftest.py),
so these run offline without downloading model weights.
"""
from __future__ import annotations

import math

import pytest

from src.search.reranker import CrossEncoderReranker, _sigmoid


@pytest.fixture
def reranker() -> CrossEncoderReranker:
    r = CrossEncoderReranker(timeout_ms=0)
    r._load_model()  # synchronous; the background thread may race otherwise
    return r


def test_sigmoid_bounds():
    assert _sigmoid(0.0) == 0.5
    assert 0.0 < _sigmoid(-10.0) < 0.01
    assert 0.99 < _sigmoid(10.0) < 1.0


def test_fake_model_loaded(reranker):
    assert reranker.model is not None
    assert reranker._model_loaded


def test_rerank_orders_by_relevance(reranker):
    candidates = [
        ("c1", "frontend designer react css", 0.9),
        ("c2", "senior python engineer aws django", 0.1),
        ("c3", "python developer", 0.5),
    ]
    ranked = reranker.rerank("senior python engineer aws", candidates, top_k=3)
    assert [pid for pid, _ in ranked] == ["c2", "c3", "c1"]
    assert all(0.0 < s < 1.0 for _, s in ranked)


def test_rerank_respects_top_k(reranker):
    candidates = [(f"c{i}", f"doc {i}", 0.0) for i in range(5)]
    assert len(reranker.rerank("doc", candidates, top_k=2)) == 2


def test_rerank_empty(reranker):
    assert reranker.rerank("anything", [], top_k=5) == []


def test_rerank_falls_back_when_model_missing(reranker):
    reranker._model = None
    ranked = reranker.rerank("q", [("a", "doc", 2.0), ("b", "doc", -2.0)], top_k=2)
    assert ranked[0] == ("a", pytest.approx(_sigmoid(2.0)))
    assert ranked[1] == ("b", pytest.approx(_sigmoid(-2.0)))


def test_rerank_falls_back_when_predict_raises(reranker):
    class Boom:
        def predict(self, *a, **kw):
            raise RuntimeError("boom")

    reranker._model = Boom()
    ranked = reranker.rerank("q", [("a", "doc", 0.0)], top_k=1)
    assert ranked == [("a", 0.5)]


def test_score_pair(reranker):
    high = reranker.score_pair("python aws", "python aws django")
    low = reranker.score_pair("python aws", "graphic designer")
    assert high > low
    assert math.isclose(reranker.score_pair("x", "y"), _sigmoid(-1.0), rel_tol=1e-6)


def test_disabled_reranker_skips_model_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core import config

    monkeypatch.setenv("CROSS_ENCODER_ENABLED", "false")
    config.get_settings.cache_clear()
    try:
        r = CrossEncoderReranker(timeout_ms=0)
    finally:
        monkeypatch.delenv("CROSS_ENCODER_ENABLED")
        config.get_settings.cache_clear()
    assert r.enabled is False
    assert r.model is None
    ranked = r.rerank("python", [("a", "doc", 2.0), ("b", "doc", -1.0)], top_k=2)
    assert [pid for pid, _ in ranked] == ["a", "b"]
