from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.core.constants import FAISS_ID_MAP_PATH, FAISS_INDEX_PATH


class VectorSearch:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.index: faiss.IndexFlatIP | None = None
        self.id_map: list[str] = []

    def build_index(self, embeddings: np.ndarray, profile_ids: list[str]) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))
        self.id_map = list(profile_ids)

    def search(self, query_embedding: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            results.append((self.id_map[int(idx)], float(score)))
        return results

    def save(
        self, index_path: Path = FAISS_INDEX_PATH, id_map_path: Path = FAISS_ID_MAP_PATH,
    ) -> None:
        if self.index is None:
            return
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(id_map_path, "w") as f:
            json.dump(self.id_map, f)

    def load(
        self, index_path: Path = FAISS_INDEX_PATH, id_map_path: Path = FAISS_ID_MAP_PATH,
    ) -> None:
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            self.dimension = self.index.d
        if id_map_path.exists():
            with open(id_map_path) as f:
                self.id_map = json.load(f)

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0
