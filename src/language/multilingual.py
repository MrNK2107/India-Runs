from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

# Imported lazily (torch alone is ~1.5 GB of commit); tests patch this name.
SentenceTransformer: Any = None

logger = logging.getLogger(__name__)


DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class MultilingualEmbedder:
    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
    ) -> None:
        # EMBEDDING_MODEL lets low-memory setups swap in another 384-dim model
        # (e.g. sentence-transformers/all-MiniLM-L6-v2). The index build and the
        # server must use the same model, so set it for both.
        self.model_name = model_name or os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        self.device = device
        self._model: Any = None
        self.dimension: int = 384

    @property
    def model(self) -> Any:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            # Respect HF_HUB_OFFLINE: if offline mode is set, only use cached files.
            # Otherwise allow downloading (e.g. first startup in HF Spaces).
            offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            cls = SentenceTransformer
            if cls is None:
                from sentence_transformers import SentenceTransformer as cls  # noqa: N813
            self._model = cls(
                self.model_name, device=self.device, local_files_only=offline,
            )
        return self._model

    def embed(self, text: str) -> np.ndarray[Any, Any]:
        result = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(result)

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray[Any, Any]:
        result = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(result)

    def cosine_similarity(self, vec_a: np.ndarray[Any, Any], vec_b: np.ndarray[Any, Any]) -> float:
        return float(np.dot(vec_a, vec_b))

    def embed_query(self, query: str) -> np.ndarray[Any, Any]:
        return self.embed(query)


def create_embedder() -> Any:
    """Embedder selected by EMBEDDING_BACKEND: ``transformer`` (default) or ``hashing``.

    ``hashing`` is the offline/lite engine: deterministic, numpy-only, no model download,
    ~0 MB of model memory. The index and the server must use the same backend.
    """
    backend = os.environ.get("EMBEDDING_BACKEND", "transformer").strip().lower()
    if backend == "hashing":
        from src.language.hashing import HashingEmbedder

        return HashingEmbedder()
    if backend not in ("", "transformer", "sentence-transformers"):
        raise ValueError(f"Unknown EMBEDDING_BACKEND {backend!r} (use 'transformer' or 'hashing')")
    return MultilingualEmbedder()
