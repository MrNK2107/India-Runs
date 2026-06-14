from __future__ import annotations

import time

from sentence_transformers import CrossEncoder

from src.core.config import get_settings


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        timeout_ms: int | None = None,
    ) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        settings = get_settings()
        self.timeout_ms = (
            timeout_ms if timeout_ms is not None else settings.cross_encoder_timeout_ms
        )

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self, query: str, candidates: list[tuple[str, str, float]], top_k: int = 10,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        pairs = [(query, doc) for _, doc, _ in candidates]
        ids = [c[0] for c in candidates]

        if self.timeout_ms > 0:
            start = time.perf_counter()
            try:
                scores = self.model.predict(pairs, show_progress_bar=False)
            except Exception:
                return [
                    (candidates[i][0], candidates[i][2])
                    for i in range(min(top_k, len(candidates)))
                ]
            elapsed_ms = (time.perf_counter() - start) * 1000
            if elapsed_ms > self.timeout_ms:
                return [
                    (candidates[i][0], candidates[i][2])
                    for i in range(min(top_k, len(candidates)))
                ]
        else:
            scores = self.model.predict(pairs, show_progress_bar=False)

        scored: list[tuple[str, float]] = []
        for pid, score in zip(ids, scores):
            scored.append((pid, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def score_pair(self, query: str, document: str) -> float:
        score = self.model.predict([(query, document)], show_progress_bar=False)
        return float(score[0])
