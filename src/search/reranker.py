from __future__ import annotations

import logging
import math
import os

# Default to offline — must be set before any sentence-transformers imports.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
        self._model = None  # type: ignore[assignment]
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
            # Temporarily allow online access for model download if needed
            old_offline = os.environ.get("HF_HUB_OFFLINE", "1")
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
                _ = self._model.predict(
                    [("warmup query", "warmup document")], show_progress_bar=False
                )
                self._model_loaded = True
                logger.info("Cross-encoder model loaded and ready")
            finally:
                # Restore offline mode for subsequent calls
                os.environ["HF_HUB_OFFLINE"] = old_offline
                os.environ["TRANSFORMERS_OFFLINE"] = old_offline
        except Exception as err:
            logger.warning(
                "Failed to load cross-encoder model '%s': %s. "
                "Reranker will fall back to hybrid RRF scores.",
                self.model_name, err,
            )
            self._model_loaded = True  # don't retry
            self._model = None

    @property
    def model(self):
        return self._model

    def rerank(  # type: ignore[return-value]
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Rerank candidates using cross-encoder. Falls back to RRF scores if model unavailable."""
        if not candidates:
            return []

        if self._model is None and not self._model_loaded:
            import threading
            for _ in range(75):
                if self._model is not None or self._model_loaded:
                    break
                threading.Event().wait(0.4)

        model = self._model
        if model is None:
            return self._fallback_score(candidates, top_k)

        pairs = [(query, doc) for _, doc, _ in candidates]
        ids = [c[0] for c in candidates]

        try:
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.warning("Cross-encoder predict failed: %s — using hybrid scores", e)
            return self._fallback_score(candidates, top_k)

        scored: list[tuple[str, float]] = []
        for pid, score in zip(ids, scores):
            scored.append((pid, _sigmoid(float(score))))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _fallback_score(
        candidates: list[tuple[str, str, float]], top_k: int,
    ) -> list[tuple[str, float]]:
        """Score candidates with sigmoid fallback when cross-encoder unavailable.

        Handles both 2-tuple (pid, score) and 3-tuple (pid, doc, score) formats.
        """
        result: list[tuple[str, float]] = []
        for c in candidates[:top_k]:
            if len(c) >= 3:
                result.append((str(c[0]), _sigmoid(float(c[2]))))
            elif len(c) >= 2:
                result.append((str(c[0]), _sigmoid(float(c[1]))))
            else:
                result.append((str(c[0]), 0.5))
        return result

    def score_pair(self, query: str, document: str) -> float:  # type: ignore[return-value]
        if self._model is None and not self._model_loaded:
            import threading
            for _ in range(75):
                if self._model is not None or self._model_loaded:
                    break
                threading.Event().wait(0.4)
        model = self._model
        if model is None:
            return 0.5
        try:
            score = model.predict([(query, document)], show_progress_bar=False)
            return _sigmoid(float(score[0]))
        except Exception:
            return 0.5
