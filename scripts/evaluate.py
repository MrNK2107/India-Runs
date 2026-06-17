from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import DATA_DIR
from src.core.constants import QUERIES_PATH, GROUND_TRUTH_PATH
from src.search.bm25_search import BM25Search
from src.search.hybrid import HybridSearch
from src.search.vector_search import VectorSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
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
        dcg += (2 ** rel - 1) / (i.bit_length()) if i > 1 else rel
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / (i.bit_length()) if i > 1 else 1.0 for i in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def cross_lingual_mrr(results: dict) -> float:
    non_en_queries = {qid for qid, q in results.get("queries", {}).items()
                      if q.get("language", "en") != "en"}
    if not non_en_queries:
        return 1.0
    mrr_sum = 0.0
    for qid in non_en_queries:
        mrr_sum += results.get("mrr", {}).get(qid, 0.0)
    return mrr_sum / len(non_en_queries)


def latency_stats(latencies: list[float]) -> dict:
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


def find_indexes() -> Path | None:
    for path in [DATA_DIR / "indexes" / "faiss_index.bin"]:
        if path.exists():
            return path.parent
    return None


def evaluate(
    queries_path: Path = QUERIES_PATH,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
) -> dict:
    index_dir = find_indexes()
    if index_dir is None:
        logger.error("No indexes found. Run 'python scripts/build_indexes.py' first.")
        return {}

    if not queries_path.exists():
        logger.warning(f"Queries file not found: {queries_path}, running demo eval on loaded profiles")
        return _demo_evaluate(index_dir)

    if not ground_truth_path.exists():
        logger.warning(f"Ground truth not found: {ground_truth_path}")
        return _demo_evaluate(index_dir)

    with open(queries_path) as f:
        queries = json.load(f)
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)
    gt_map = ground_truth if isinstance(ground_truth, dict) else {}

    vector_search = VectorSearch()
    vector_search.load(index_dir / "faiss_index.bin", index_dir / "faiss_id_map.json")
    bm25_search = BM25Search()
    bm25_search.load(index_dir / "bm25_index.pkl")

    from src.language.multilingual import MultilingualEmbedder
    embedder = MultilingualEmbedder()
    hybrid = HybridSearch(vector_search, bm25_search, embedder)

    all_metrics: dict[str, list] = {
        "p@5": [], "p@10": [], "p@20": [],
        "r@5": [], "r@10": [], "r@20": [],
        "mrr": [],
        "ndcg@10": [],
        "latencies": [],
    }

    queries_list = queries if isinstance(queries, list) else list(queries.values())

    for q in queries_list:
        query_text = q.get("query", q.get("raw_query", ""))
        qid = q.get("query_id", q.get("id", ""))
        relevant = set(q.get("relevant_ids", q.get("profile_ids", gt_map.get(qid, []))))
        if not query_text or not relevant:
            continue

        start = time.perf_counter()
        results = hybrid.search(query_text, top_k=50)
        elapsed = (time.perf_counter() - start) * 1000
        all_metrics["latencies"].append(elapsed)

        retrieved = [pid for pid, _ in results]

        for k in (5, 10, 20):
            all_metrics[f"p@{k}"].append(precision_at_k(retrieved, relevant, k))
            all_metrics[f"r@{k}"].append(recall_at_k(retrieved, relevant, k))

        all_metrics["mrr"].append(mean_reciprocal_rank(retrieved, relevant))
        all_metrics["ndcg@10"].append(ndcg_at_k(retrieved, relevant, 10))

    summary = {}
    for metric, values in all_metrics.items():
        if values:
            summary[metric] = {
                "mean": mean(values),
                "median": median(values),
                "min": min(values),
                "max": max(values),
            }
        else:
            summary[metric] = {}

    summary["latency"] = latency_stats(all_metrics["latencies"])
    summary["total_queries"] = len(all_metrics["mrr"])
    summary["cross_lingual_mrr"] = cross_lingual_mrr({
        "queries": {i: q for i, q in enumerate(queries_list)},
        "mrr": {i: v for i, v in enumerate(all_metrics["mrr"])},
    })

    logger.info("Evaluation results:")
    for metric, vals in summary.items():
        if isinstance(vals, dict) and "mean" in vals:
            logger.info(f"  {metric}: mean={vals['mean']:.4f}, median={vals['median']:.4f}")

    return summary


def _demo_evaluate(index_dir: Path) -> dict:
    logger.info("Running demo evaluation with 5 sample queries")
    queries = [
        "Senior Python developer with Django experience",
        "DevOps engineer AWS Kubernetes",
        "Frontend engineer React TypeScript",
        "Data scientist machine learning Python",
        "Product manager B2B SaaS",
    ]

    vector_search = VectorSearch()
    vector_search.load(index_dir / "faiss_index.bin", index_dir / "faiss_id_map.json")
    bm25_search = BM25Search()
    bm25_search.load(index_dir / "bm25_index.pkl")

    from src.language.multilingual import MultilingualEmbedder
    embedder = MultilingualEmbedder()
    hybrid = HybridSearch(vector_search, bm25_search, embedder)

    all_metrics: dict[str, list] = {
        "p@5": [], "p@10": [], "r@5": [], "r@10": [],
        "mrr": [], "ndcg@10": [], "latencies": [],
    }

    for query_text in queries:
        start = time.perf_counter()
        results = hybrid.search(query_text, top_k=20)
        elapsed = (time.perf_counter() - start) * 1000
        all_metrics["latencies"].append(elapsed)

        retrieved = [pid for pid, _ in results]
        logger.info(f"  Query: {query_text[:50]}... -> {len(results)} results")

    summary = {
        "total_queries": len(queries),
        "total_results": 0,
        "latency": latency_stats(all_metrics["latencies"]),
        "note": "Demo mode: queries executed, no ground truth for precision metrics",
    }

    logger.info("Demo evaluation complete:")
    logger.info(f"  Queries: {summary['total_queries']}")
    logger.info(f"  Latency: p50={summary['latency']['p50']:.0f}ms, p95={summary['latency']['p95']:.0f}ms")
    return summary


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, indent=2))
