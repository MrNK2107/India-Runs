from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MultilingualEmbedder:
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None
        self.dimension: int = 384

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        result = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(result)

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        result = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(result)

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed(query)
