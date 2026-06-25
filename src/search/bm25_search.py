from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.core.constants import BM25_INDEX_PATH


class BM25Search:
    def __init__(self) -> None:
        self.index: BM25Okapi | None = None
        self.id_map: list[str] = []
        self.corpus_tokenized: list[list[str]] = []

    def build_index(self, documents: list[str], profile_ids: list[str]) -> None:
        self.corpus_tokenized = [self._tokenize(doc) for doc in documents]
        self.index = BM25Okapi(self.corpus_tokenized)
        self.id_map = list(profile_ids)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if self.index is None:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        n_candidates = len(scores)
        if n_candidates <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            partition_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = partition_indices[np.argsort(-scores[partition_indices])]

        results: list[tuple[str, float]] = []
        for idx in top_indices:
            if scores[idx] < 0:
                continue
            results.append((self.id_map[int(idx)], float(scores[idx])))
        return results

    def save(self, path: Path = BM25_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus_tokenized": self.corpus_tokenized,
            "id_map": self.id_map,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: Path = BM25_INDEX_PATH) -> None:
        if not path.exists():
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.corpus_tokenized = data["corpus_tokenized"]
        self.id_map = data["id_map"]
        self.index = BM25Okapi(self.corpus_tokenized)

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    @property
    def size(self) -> int:
        return len(self.id_map)
