from __future__ import annotations

import logging
import random
import time

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

# ISO 639-1 codes supported by Google Translate
SUPPORTED_LANGUAGES = {
    "hi": "hindi", "ta": "tamil", "te": "telugu", "mr": "marathi",
    "bn": "bengali", "kn": "kannada", "ml": "malayalam", "gu": "gujarati",
    "pa": "punjabi", "or": "odia", "as": "assamese",
    "ur": "urdu", "sd": "sindhi", "ks": "kashmiri", "ne": "nepali",
}


class TranslationPipeline:
    def __init__(self) -> None:
        self._translator: GoogleTranslator | None = None

    def _get_translator(self) -> GoogleTranslator:
        if self._translator is None:
            self._translator = GoogleTranslator(source="auto", target="en")
        return self._translator

    def translate_to_english(
        self, text: str, source_lang: str, is_batch_call: bool = False,
    ) -> dict[str, str | float | bool]:
        if source_lang == "en":
            return {
                "original": text,
                "translated": text,
                "confidence": 1.0,
                "model_used": "none",
                "translation_fallback": False,
            }

        if not is_batch_call:
            time.sleep(random.uniform(0.1, 0.5))

        try:
            translator = self._get_translator()
            translated_text = translator.translate(text)
            return {
                "original": text,
                "translated": translated_text or text,
                "confidence": 0.85,
                "model_used": "GoogleTranslate",
                "translation_fallback": False,
            }

        except Exception as e:
            logger.warning(f"Translation failed for {source_lang}: {e}")
            return {
                "original": text,
                "translated": text,
                "confidence": 0.0,
                "model_used": "none",
                "translation_fallback": True,
            }

    def translate_batch(
        self, texts: list[tuple[str, str]]
    ) -> list[dict]:
        results: list[dict] = []
        for i, (text, lang) in enumerate(texts):
            if i > 0 and i % 10 == 0:
                time.sleep(0.3)
            results.append(self.translate_to_english(text, lang, is_batch_call=True))
        return results
