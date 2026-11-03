from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import (
    cross_lingual_mrr,
    latency_stats,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


# -------------------------------------------------------------------- NDCG


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)


def test_ndcg_uses_log2_discount() -> None:
    # One relevant doc at rank 2 of 3, one relevant overall: DCG = 1/log2(3), IDCG = 1.
    assert ndcg_at_k(["x", "a", "y"], {"a"}, 3) == pytest.approx(1 / math.log2(3))
    # Ranks 1 and 4 relevant, two relevant overall.
    got = ndcg_at_k(["a", "x", "y", "b"], {"a", "b"}, 4)
    assert got == pytest.approx(_dcg([1, 0, 0, 1]) / _dcg([1, 1]))


def test_ndcg_duplicates_do_not_exceed_one() -> None:
    got = ndcg_at_k(["a", "a", "a"], {"a", "b"}, 3)
    assert 0.0 <= got <= 1.0
    assert got == pytest.approx(_dcg([1]) / _dcg([1, 1]))


def test_ndcg_edge_cases() -> None:
    assert ndcg_at_k([], {"a"}, 5) == 0.0
    assert ndcg_at_k(["a"], set(), 5) == 0.0
    assert ndcg_at_k(["a"], {"a"}, 0) == 0.0
    assert ndcg_at_k(["x", "a"], {"a"}, 1) == 0.0


# -------------------------------------------------------- precision / recall


def test_precision_and_recall_basic() -> None:
    retrieved = ["a", "x", "b", "y"]
    assert precision_at_k(retrieved, {"a", "b"}, 2) == pytest.approx(0.5)
    assert precision_at_k(retrieved, {"a", "b"}, 4) == pytest.approx(0.5)
    assert recall_at_k(retrieved, {"a", "b", "c", "d"}, 3) == pytest.approx(0.5)


def test_precision_short_list_divides_by_returned() -> None:
    # Documented behaviour: fewer than k results -> precision over what was returned.
    assert precision_at_k(["a"], {"a"}, 10) == pytest.approx(1.0)


def test_duplicates_are_counted_once() -> None:
    assert recall_at_k(["a", "a"], {"a", "b"}, 2) == pytest.approx(0.5)
    assert precision_at_k(["a", "a"], {"a", "b"}, 2) <= 0.5


def test_precision_recall_edge_cases() -> None:
    assert precision_at_k([], {"a"}, 5) == 0.0
    assert precision_at_k(["a"], {"a"}, 0) == 0.0
    assert recall_at_k(["a"], set(), 5) == 0.0
    assert recall_at_k(["a"], {"a"}, 0) == 0.0


# ----------------------------------------------------------------------- MRR


def test_mrr() -> None:
    assert mean_reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert mean_reciprocal_rank(["a"], {"a"}) == 1.0
    assert mean_reciprocal_rank(["x"], {"a"}) == 0.0
    assert mean_reciprocal_rank([], {"a"}) == 0.0


def test_cross_lingual_mrr() -> None:
    results = {
        "queries": {"q1": {"language": "hi"}, "q2": {"language": "en"}, "q3": {"language": "ta"}},
        "mrr": {"q1": 1.0, "q2": 0.0, "q3": 0.5},
    }
    assert cross_lingual_mrr(results) == pytest.approx(0.75)
    # Missing per-query MRR counts as 0.
    assert cross_lingual_mrr({"queries": {"q1": {"language": "hi"}}}) == 0.0


# ------------------------------------------------------------------- latency


def test_latency_stats() -> None:
    lat = [float(i) for i in range(1, 101)]
    s = latency_stats(lat)
    assert s["min"] == 1.0 and s["max"] == 100.0 and s["mean"] == pytest.approx(50.5)
    assert s["p50"] <= s["p95"] <= s["p99"] <= s["max"]
    assert latency_stats([]) == {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "min": 0, "max": 0}
    assert latency_stats([7.0])["p99"] == 7.0


def test_evaluate_script_uses_shared_metrics() -> None:
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_script", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.ndcg_at_k is ndcg_at_k
