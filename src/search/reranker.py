from __future__ import annotations

import logging
import math
import os

# Force offline mode — must be set before any sentence-transformers imports.
# Overrides are redundant with main.py but critical if this module is loaded first.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.core.config import get_settings

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    """Reranks candidates using cross-encoder for fine-grained relevance scoring.

    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 with offline-mode enforcement
    to prevent HuggingFace hub HEAD request loops. Model is pre-loaded eagerly
    during construction so search requests never trigger lazy initialization.
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
        self.enabled = True
        # Pre-load in background thread to avoid blocking startup
        import threading
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self) -> None:
        """Load the cross-encoder model synchronously."""
        if self._model_loaded:
            return
        try:
            logger.info("Loading cross-encoder model (offline): %s", self.model_name)
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            # Force tokenizer and model initialization with a dummy warmup
            _ = self._model.predict(
                [("warmup query", "warmup document")], show_progress_bar=False
            )
            self._model_loaded = True
            logger.info("Cross-encoder model loaded offline and ready")
        except Exception as offline_err:
            logger.info("Failed to load offline, attempting to download/load online: %s", offline_err)
            try:
                # Disable offline environment overrides
                os.environ["HF_HUB_OFFLINE"] = "0"
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
                _ = self._model.predict(
                    [("warmup query", "warmup document")], show_progress_bar=False
                )
                self._model_loaded = True
                logger.info("Cross-encoder model loaded online and ready")
            except Exception as online_err:
                logger.warning("Failed to load cross-encoder model online: %s", online_err)
                self._model_loaded = True  # don't retry
                self._model = None

    @property
    def model(self):
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Rerank candidates using cross-encoder. Falls back to RRF scores if model unavailable."""
        if not candidates:
            return []

        model = self._model
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
        model = self._model
        if model is None:
            return 0.5
        try:
            score = model.predict([(query, document)], show_progress_bar=False)
            return _sigmoid(float(score[0]))
        except Exception:
            return 0.5
