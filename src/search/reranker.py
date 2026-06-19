from __future__ import annotations

import logging
import math
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.core.config import get_settings

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    """Reranks candidates using hybrid RRF scores.

    Cross-encoder reranking is disabled by default due to a
    sentence-transformers 3.x bug causing infinite HuggingFace hub
    HEAD request loops. Falls back to sigmoid-normalized hybrid
    search scores.

    Set `cross_encoder_enabled: true` in config to attempt cross-encoder
    loading, or set env `CROSS_ENCODER_ENABLED=true`.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        timeout_ms: int | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = None
        self._model_loaded = False
        settings = get_settings()
        self.timeout_ms = (
            timeout_ms if timeout_ms is not None else settings.cross_encoder_timeout_ms
        )
        self.enabled = os.environ.get("CROSS_ENCODER_ENABLED", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        if self.enabled:
            logger.info("Cross-encoder is ENABLED — will attempt model loading on first use")

    def _try_load_model(self):
        """Attempt to load cross-encoder, return None if it fails."""
        if self._model_loaded:
            return self._model
        if not self.enabled:
            self._model_loaded = True
            logger.info("Cross-encoder disabled — skipping model load")
            return None
        try:
            logger.info("Loading cross-encoder model: %s", self.model_name)
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            # Force tokenizer initialization with a dummy predict
            _ = self._model.predict(
                [("test query", "test document")], show_progress_bar=False
            )
            self._model_loaded = True
            logger.info("Cross-encoder model loaded and ready")
            return self._model
        except Exception as e:
            logger.warning("Failed to load cross-encoder model: %s", e)
            self._model_loaded = True  # don't retry
            self._model = None
            return None

    @property
    def model(self):
        return self._try_load_model()

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Rerank candidates. Falls back to RRF scores if cross-encoder unavailable."""
        if not candidates:
            return []

        model = self._try_load_model()
        if model is None:
            return [(c[0], _sigmoid(c[2])) for c in candidates[:top_k]]

        pairs = [(query, doc) for _, doc, _ in candidates]
        ids = [c[0] for c in candidates]

        try:
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.warning("Cross-encoder predict failed: %s — using hybrid scores", e)
            return [(c[0], _sigmoid(c[2])) for c in candidates[:top_k]]

        scored: list[tuple[str, float]] = []
        for pid, score in zip(ids, scores):
            scored.append((pid, _sigmoid(float(score))))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def score_pair(self, query: str, document: str) -> float:
        model = self._try_load_model()
        if model is None:
            return 0.5
        try:
            score = model.predict([(query, document)], show_progress_bar=False)
            return _sigmoid(float(score[0]))
        except Exception:
            return 0.5
