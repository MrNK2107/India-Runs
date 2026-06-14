from __future__ import annotations

from src.api.routes.health import init_health
from src.core.models import SearchRequest
from src.fairness.bias_detector import BiasDetector
from src.fairness.metrics import compute_all_fairness_metrics, compute_demographic_parity


def test_health_init():
    init_health(index_size=100)
    from src.api.routes.health import _index_size
    assert _index_size == 100


def test_search_endpoint_schema():
    req = SearchRequest(query="Python developer", max_results=10)
    assert req.query == "Python developer"
    assert req.max_results == 10


def test_bias_detector_no_matches():
    detector = BiasDetector()
    result = detector.check_name_bias([], {})
    assert not result["detected"]


def test_bias_detector_name(sample_profile):
    detector = BiasDetector()
    from src.core.models import MatchMetadata, MatchResult, MatchScores, SearchMethod
    match = MatchResult(
        query_id="q1", profile_id="test-001", rank=1, name="Test User",
        scores=MatchScores(overall=0.9, semantic_similarity=0.8, keyword_match=0.7,
                           skill_match=0.9, experience_match=0.6, confidence=0.8),
        metadata=MatchMetadata(search_method=SearchMethod.HYBRID),
    )
    result = detector.check_name_bias([match], {"test-001": sample_profile})
    assert "observations" in result


def test_demographic_parity_no_matches():
    result = compute_demographic_parity([], {}, "university")
    assert result == 1.0


def test_all_fairness_metrics(sample_profile):
    from src.core.models import MatchMetadata, MatchResult, MatchScores, SearchMethod
    match = MatchResult(
        query_id="q1", profile_id="test-001", rank=1, name="User",
        scores=MatchScores(overall=0.8, semantic_similarity=0.8, keyword_match=0.7,
                           skill_match=0.9, experience_match=0.6, confidence=0.8),
        metadata=MatchMetadata(search_method=SearchMethod.HYBRID),
    )
    result = compute_all_fairness_metrics([match], {"test-001": sample_profile})
    assert "demographic_parity" in result
    assert "language_bias" in result
