from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TranslationPipeline:
    def __init__(
        self,
        primary_model: str = "Helsinki-NLP/opus-mt-mul",
        fallback_model: str = "facebook/mbart-large-50-many-to-many-mmt",
    ) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self._primary: object | None = None
        self._fallback: object | None = None

    def load_models(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self._primary is None:
            logger.info(f"Loading primary translation model: {self.primary_model}")
            self._primary = AutoTokenizer.from_pretrained(self.primary_model)
            self._fallback = AutoModelForSeq2SeqLM.from_pretrained(self.primary_model)

    def translate_to_english(
        self, text: str, source_lang: str
    ) -> dict[str, str | float]:
        if source_lang == "en":
            return {
                "original": text,
                "translated": text,
                "confidence": 1.0,
                "model_used": "none",
            }

        return {
            "original": text,
            "translated": text,
            "confidence": 0.0,
            "model_used": "none",
        }

    def translate_batch(
        self, texts: list[tuple[str, str]]
    ) -> list[dict]:
        return [self.translate_to_english(t, lang) for t, lang in texts]

    def _get_model_name(self, source_lang: str) -> str | None:
        pair_map = {
            "hi": "Helsinki-NLP/opus-mt-hi-en",
            "ta": "Helsinki-NLP/opus-mt-ta-en",
            "te": "Helsinki-NLP/opus-mt-te-en",
            "bn": "Helsinki-NLP/opus-mt-bn-en",
            "mr": "Helsinki-NLP/opus-mt-mr-en",
            "gu": "Helsinki-NLP/opus-mt-gu-en",
            "pa": "Helsinki-NLP/opus-mt-pa-en",
            "kn": "Helsinki-NLP/opus-mt-kn-en",
            "ml": "Helsinki-NLP/opus-mt-ml-en",
        }
        return pair_map.get(source_lang)
