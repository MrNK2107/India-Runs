from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.core.constants import BM25_INDEX_PATH

logger = logging.getLogger(__name__)


class BM25Search:
    def __init__(self) -> None:
        self.index: BM25Okapi | None = None
        self.id_map: list[str] = []
        self.corpus_tokenized: list[list[str]] = []
        self._load_path: Path | None = None
        self._load_event = threading.Event()
        self._load_event.set()  # not waiting by default; cleared only by lazy_load
        self._load_error: Exception | None = None
        self._loading = False

    def build_index(self, documents: list[str], profile_ids: list[str]) -> None:
        self.corpus_tokenized = [self._tokenize(doc) for doc in documents]
        self.index = BM25Okapi(self.corpus_tokenized)
        self.id_map = list(profile_ids)
        self._load_event.set()

    def lazy_load(self, path: Path) -> None:
        """Start loading BM25 index in a background thread.

        The first call to search() will block until loading is complete.
        Use wait_ready() to block explicitly before search.
        """
        if self.index is not None or self._loading:
            return  # already loaded or loading
        self._loading = True
        logger.info("Starting background BM25 index load from %s", path)
        self._load_path = path
        self._load_event.clear()
        thread = threading.Thread(target=self._load_async, daemon=True)
        thread.start()

    def _load_async(self) -> None:
        """Background thread: load the BM25 index from disk."""
        try:
            self.load(self._load_path)
            logger.info("BM25 index loaded in background (%d docs)", self.size)
        except Exception as e:
            self._load_error = e
            logger.warning("BM25 background load failed: %s", e)
        finally:
            self._loading = False
            self._load_event.set()

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Wait for the BM25 index to finish loading.

        Args:
            timeout: Max seconds to wait. None = wait indefinitely.

        Returns:
            True if loaded, False if timeout expired.
        """
        if self.index is not None:
            return True
        self._load_event.wait(timeout=timeout)
        return self.index is not None

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if self.index is None:
            self._load_event.wait(timeout=30.0)
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
