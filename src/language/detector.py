from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0


class LanguageDetector:
    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence

    def detect(self, text: str) -> dict[str, str | bool]:
        try:
            lang = detect(text)
        except LangDetectException:
            lang = "en"

        return {
            "language": lang,
            "is_english": lang == "en",
            "needs_translation": lang != "en",
        }

    def detect_batch(self, texts: list[str]) -> list[dict[str, str | bool]]:
        return [self.detect(t) for t in texts]
