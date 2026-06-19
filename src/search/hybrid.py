from __future__ import annotations

import hashlib
import time
from collections import OrderedDict

from src.core.config import get_scoring_config
from src.language.multilingual import MultilingualEmbedder
from src.search.bm25_search import BM25Search
from src.search.vector_search import VectorSearch


class SearchCache:
    def __init__(self, maxsize: int = 256, ttl_seconds: int = 60) -> None:
        self._cache: OrderedDict[str, tuple[float, list[tuple[str, float]]]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds

    def _make_key(self, query: str, top_k: int) -> str:
        raw = f"{query}::top_k={top_k}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, query: str, top_k: int) -> list[tuple[str, float]] | None:
        key = self._make_key(query, top_k)
        if key not in self._cache:
            return None
        timestamp, results = self._cache[key]
        if time.monotonic() - timestamp > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return results

    def set(self, query: str, top_k: int, results: list[tuple[str, float]]) -> None:
        key = self._make_key(query, top_k)
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = (time.monotonic(), results)

    def clear(self) -> None:
        self._cache.clear()


class HybridSearch:
    def __init__(
        self, vector_search: VectorSearch, bm25_search: BM25Search, embedder: MultilingualEmbedder,
    ) -> None:
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.embedder = embedder
        self.rrf_k = get_scoring_config().get("rrf_k", 60)
        self._cache = SearchCache()

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        cached = self._cache.get(query, top_k)
        if cached is not None:
            return cached

        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_search.search(query_embedding, top_k=top_k)
        bm25_results = self.bm25_search.search(query, top_k=top_k)

        ranks: list[list[tuple[str, float]]] = []
        if vector_results:
            ranks.append(vector_results)
        if bm25_results:
            ranks.append(bm25_results)

        if not ranks:
            return []

        results = self.reciprocal_rank_fusion(ranks, k=self.rrf_k)
        self._cache.set(query, top_k, results)
        return results

    def reciprocal_rank_fusion(
        self, rankings: list[list[tuple[str, float]]], k: int = 60,
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
