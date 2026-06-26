from __future__ import annotations

from statistics import mean
from typing import Any


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len([doc for doc in top_k if doc in relevant]) / len(top_k)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = len([doc for doc in top_k if doc in relevant])
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    dcg = 0.0
    for i, doc in enumerate(top_k, start=1):
        rel = 1.0 if doc in relevant else 0.0
        dcg += (2**rel - 1) / (i.bit_length()) if i > 1 else rel
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / (i.bit_length()) if i > 1 else 1.0 for i in range(1, ideal + 1))
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
