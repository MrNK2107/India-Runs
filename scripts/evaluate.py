from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import DATA_DIR
from src.core.constants import GROUND_TRUTH_PATH, QUERIES_PATH
from src.search.bm25_search import BM25Search
from src.search.hybrid import HybridSearch
from src.search.vector_search import VectorSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        dcg += (2 ** rel - 1) / (i.bit_length()) if i > 1 else rel
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / (i.bit_length()) if i > 1 else 1.0 for i in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def cross_lingual_mrr(results: dict) -> float:
    non_en_queries = {
        qid for qid, q in results.get("queries", {}).items()
        if q.get("language", "en") != "en"
    }
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


def find_index_dir() -> Path | None:
    path = DATA_DIR / "indexes" / "faiss_index.bin"
    return path.parent if path.exists() else None


def evaluate(
    queries_path: Path = QUERIES_PATH,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
) -> dict:
    index_dir = find_index_dir()
    if index_dir is None:
        logger.error("No indexes found. Run 'python scripts/build_indexes.py --sample 50' first.")
        return {}

    errors: list[str] = []
    if not queries_path.exists():
        errors.append(f"Queries file not found: {queries_path}")
    if not ground_truth_path.exists():
        errors.append(f"Ground truth file not found: {ground_truth_path}")
    if errors:
        for e in errors:
            logger.error(e)
        return {}

    with open(queries_path) as f:
        queries_raw = json.load(f)
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    queries_list = queries_raw if isinstance(queries_raw, list) else list(queries_raw.values())
    gt_map = ground_truth if isinstance(ground_truth, dict) else {}

    logger.info(f"Loaded {len(queries_list)} queries and {len(gt_map)} ground truth entries")

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

    evaluated = 0
    skipped = 0

    for q in queries_list:
        query_text = q.get("query", q.get("raw_query", ""))
        qid = q.get("query_id", q.get("id", ""))
        relevant = set(gt_map.get(qid, []))
        if not query_text or not relevant:
            skipped += 1
            continue

        t0 = time.perf_counter()
        results = hybrid.search(query_text, top_k=50)
        elapsed = (time.perf_counter() - t0) * 1000
        all_metrics["latencies"].append(elapsed)

        retrieved = [pid for pid, _ in results]

        for k in (5, 10, 20):
            all_metrics[f"p@{k}"].append(precision_at_k(retrieved, relevant, k))
            all_metrics[f"r@{k}"].append(recall_at_k(retrieved, relevant, k))

        all_metrics["mrr"].append(mean_reciprocal_rank(retrieved, relevant))
        all_metrics["ndcg@10"].append(ndcg_at_k(retrieved, relevant, 10))
        evaluated += 1

    if evaluated == 0:
        logger.error("No queries could be evaluated (check ground truth IDs match query IDs)")
        return {}

    summary: dict = {}
    for metric, values in all_metrics.items():
        if values:
            summary[metric] = {
                "mean": round(mean(values), 4),
                "median": round(median(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
        else:
            summary[metric] = {}

    summary["latency"] = latency_stats(all_metrics["latencies"])
    for k in ("p50", "p95", "p99", "mean", "min", "max"):
        if k in summary["latency"]:
            summary["latency"][k] = round(summary["latency"][k], 1)

    summary["total_queries"] = evaluated
    summary["skipped"] = skipped
    summary["cross_lingual_mrr"] = round(cross_lingual_mrr({
        "queries": {i: q for i, q in enumerate(queries_list)},
        "mrr": {i: v for i, v in enumerate(all_metrics["mrr"])},
    }), 4)

    logger.info("Evaluation results:")
    for metric in ("p@5", "p@10", "r@5", "r@10", "mrr", "ndcg@10"):
        if metric in summary:
            s = summary[metric]
            logger.info(f"  {metric}: mean={s['mean']:.4f}, median={s['median']:.4f}")
    lat = summary["latency"]
    logger.info(f"  latency: p50={lat['p50']:.0f}ms, p95={lat['p95']:.0f}ms")
    logger.info(f"  cross-lingual MRR: {summary['cross_lingual_mrr']:.4f}")

    return summary


if __name__ == "__main__":
    result = evaluate()
    if result:
        report_path = DATA_DIR / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Report saved to {report_path}")
        print(json.dumps(result, indent=2))
    else:
        logger.error("Evaluation failed")
        sys.exit(1)
