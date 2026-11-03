from __future__ import annotations

import math
from statistics import mean
from typing import Any


def _unique(docs: list[str]) -> list[str]:
    """First occurrence of each id; a duplicate in a ranking must not count twice."""
    return list(dict.fromkeys(docs))


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Relevant share of the top k; with fewer than k results, of what was returned."""
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in _unique(top_k) if doc in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    hits = sum(1 for doc in _unique(retrieved[:k]) if doc in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k with the standard 1/log2(rank + 1) discount.

    (Previously used int.bit_length() as the discount, which is not log2 and over-weighted
    rank 2, and counted duplicate ids repeatedly, so scores could exceed 1.)
    """
    if k <= 0 or not relevant:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant and doc not in seen:
            dcg += 1.0 / math.log2(rank + 1)
        seen.add(doc)
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def cross_lingual_mrr(results: dict[str, Any]) -> float:
    non_en_queries = {
        qid
        for qid, q in results.get("queries", {}).items()
        if q.get("language", "en") != "en"
    }
    if not non_en_queries:
        return 1.0
    mrr_sum = 0.0
    for qid in non_en_queries:
        mrr_sum += results.get("mrr", {}).get(qid, 0.0)
    return mrr_sum / len(non_en_queries)


def latency_stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "min": 0, "max": 0}
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return {
        "p50": sorted_lat[int(n * 0.50)],
        "p95": sorted_lat[int(n * 0.95)],
        "p99": sorted_lat[int(n * 0.99)],
        "mean": mean(latencies),
        "min": min(latencies),
        "max": max(latencies),
    }


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "cross_lingual_mrr",
    "latency_stats",
]
