from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.core.constants import FAISS_ID_MAP_PATH, FAISS_INDEX_PATH

logger = logging.getLogger(__name__)


class VectorSearch:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.index: faiss.Index | None = None
        # Position i in the FAISS index -> profile id. ``None`` marks a
        # tombstoned vector (superseded by a re-ingested profile): HNSW
        # indexes can't delete, so stale vectors are skipped at search time.
        self.id_map: list[str | None] = []

    def build_index(self, embeddings: np.ndarray[Any, Any], profile_ids: list[str]) -> None:
        expected_dim = self.dimension
        actual_dim = embeddings.shape[1] if embeddings.ndim > 1 else embeddings.shape[0]
        if actual_dim != expected_dim:
            logger.error(
                "Dimension mismatch in build_index: expected %d, got %d. "
                "Set dimension=%d or fix embedder output.",
                expected_dim, actual_dim, actual_dim,
            )
            raise ValueError(
                f"Embedding dimension {actual_dim} does not match "
                f"VectorSearch dimension {expected_dim}"
            )
        self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.add(embeddings.astype(np.float32))
        self.id_map = list(profile_ids)

    def upsert(self, embeddings: np.ndarray[Any, Any], profile_ids: list[str]) -> None:
        """Add vectors for ``profile_ids``, replacing any existing ones.

        Creates an empty index on first use. Previous vectors of the same ids
        are tombstoned (their id_map slot set to None) so searches stop
        returning them; the vectors themselves stay in the HNSW graph.
        """
        if embeddings.ndim != 2 or embeddings.shape[0] != len(profile_ids):
            raise ValueError(
                f"Expected {len(profile_ids)} embeddings of shape (n, d), "
                f"got {embeddings.shape}"
            )
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("Duplicate profile ids in one upsert")
        if not profile_ids:
            return
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"VectorSearch dimension {self.dimension}"
            )
        if self.index is None:
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        # Keep positions aligned with the index even if a loaded id_map was short.
        if len(self.id_map) < self.index.ntotal:
            self.id_map.extend([None] * (self.index.ntotal - len(self.id_map)))
        replaced = set(profile_ids)
        for i, pid in enumerate(self.id_map):
            if pid is not None and pid in replaced:
                self.id_map[i] = None
        self.index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self.id_map.extend(profile_ids)

    @property
    def _tombstones(self) -> int:
        return sum(1 for pid in self.id_map if pid is None)

    def search(
        self, query_embedding: np.ndarray[Any, Any], top_k: int = 50
    ) -> list[tuple[str, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        if query_vec.shape[1] != self.dimension:
            logger.error(
                "Dimension mismatch in search: index dimension=%d, "
                "query dimension=%d. Returning empty results.",
                self.dimension, query_vec.shape[1],
            )
            return []
        # Over-fetch by the number of tombstoned slots so stale vectors don't
        # crowd live ones out of the top_k.
        fetch_k = min(top_k + self._tombstones, self.index.ntotal)
        try:
            scores, indices = self.index.search(query_vec, fetch_k)
        except Exception as e:
            logger.error("FAISS search failed: %s", e)
            return []
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            pid = self.id_map[int(idx)]
            if pid is None:
                continue
            results.append((pid, float(score)))
            if len(results) >= top_k:
                break
        return results

    def save(
        self, index_path: Path = FAISS_INDEX_PATH, id_map_path: Path = FAISS_ID_MAP_PATH,
    ) -> None:
        if self.index is None:
            return
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(id_map_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f)

    def load(
        self, index_path: Path = FAISS_INDEX_PATH, id_map_path: Path = FAISS_ID_MAP_PATH,
    ) -> None:
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            self.dimension = self.index.d
        if id_map_path.exists():
            with open(id_map_path, encoding="utf-8") as f:
                self.id_map = json.load(f)
        elif self.index is not None:
            logger.warning(
                "FAISS index loaded from %s but id_map missing at %s; "
                "searches may return empty or incorrect results",
                index_path, id_map_path,
            )

    @property
    def size(self) -> int:
        """Number of live (non-tombstoned) vectors."""
        if not self.index:
            return 0
        return self.index.ntotal - self._tombstones
