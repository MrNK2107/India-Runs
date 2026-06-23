from __future__ import annotations

from src.ingestion.quality_scorer import compute_data_quality_score


def test_full_profile_quality(sample_profile):
    score = compute_data_quality_score(sample_profile)
    assert score > 0.5


def test_cross_encoder_reranker_imports(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass
        def predict(self, pairs, **kwargs):
            return [0.5] * len(pairs)

    monkeypatch.setattr("sentence_transformers.CrossEncoder", FakeCrossEncoder)
    from src.search.reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker(timeout_ms=100)
    assert reranker.timeout_ms == 100


def test_orchestrator_imports():
    from src.agents.orchestrator import Orchestrator
    assert Orchestrator.__name__ == "Orchestrator"


def test_models_exist():
    from src.core.models import (
        Profile,
        Rationale,
        SearchResponse,
    )
    assert Profile.__name__ == "Profile"
    assert SearchResponse.__name__ == "SearchResponse"
    assert Rationale.__name__ == "Rationale"


def test_evaluation_metrics():
    from src.evaluation.metrics import precision_at_k, recall_at_k, mean_reciprocal_rank

    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c", "f"}
    p5 = precision_at_k(retrieved, relevant, 5)
    assert p5 == 0.4
    r5 = recall_at_k(retrieved, relevant, 5)
    assert r5 > 0
    mrr = mean_reciprocal_rank(retrieved, relevant)
    assert mrr == 1.0
