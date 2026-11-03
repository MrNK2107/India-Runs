from __future__ import annotations

import logging
import random
import time
from typing import Any

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

# ISO 639-1 codes supported by Google Translate
SUPPORTED_LANGUAGES = {
    "hi": "hindi", "ta": "tamil", "te": "telugu", "mr": "marathi",
    "bn": "bengali", "kn": "kannada", "ml": "malayalam", "gu": "gujarati",
    "pa": "punjabi", "or": "odia", "as": "assamese",
    "ur": "urdu", "sd": "sindhi", "ks": "kashmiri", "ne": "nepali",
}

DEFAULT_MBART_MODEL = "facebook/mbart-large-50-many-to-many-mmt"


def mbart_model_name() -> str:
    """Fallback translation model from settings.yaml (models.translation.fallback)."""
    try:
        from src.core import config

        cfg = config.get_app_config()
        name = cfg.get("models", {}).get("translation", {}).get("fallback")
    except Exception:
        return DEFAULT_MBART_MODEL
    return name if isinstance(name, str) and name else DEFAULT_MBART_MODEL


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
            logger.warning(
                "Google Translation failed for %s, trying MBART: %s",
                source_lang, e
            )
            mbart_name = mbart_model_name()

            try:
                import transformers

                tok_cls: Any = getattr(transformers, "MBart50TokenizerFast")
                model_cls: Any = transformers.MBartForConditionalGeneration

                if not hasattr(self, "_mbart_tokenizer") or self._mbart_tokenizer is None:
                    logger.info(f"Loading MBART model: {mbart_name}")
                    self._mbart_tokenizer: Any = tok_cls.from_pretrained(mbart_name)
                    self._mbart_model: Any = (
                        model_cls.from_pretrained(
                            mbart_name
                        )
                    )

                mbart_lang_map = {
                    "hi": "hi_IN", "ta": "ta_IN", "te": "te_IN", "mr": "mr_IN",
                    "bn": "bn_IN", "kn": "kn_IN", "ml": "ml_IN", "gu": "gu_IN",
                    "pa": "pa_IN", "ur": "ur_PK",
                }
                mbart_code = mbart_lang_map.get(source_lang, "hi_IN")

                self._mbart_tokenizer.src_lang = mbart_code
                encoded = self._mbart_tokenizer(text, return_tensors="pt")
                generated_tokens = self._mbart_model.generate(
                    **encoded,
                    forced_bos_token_id=self._mbart_tokenizer.lang_code_to_id["en_XX"]
                )
                translated_text = self._mbart_tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0]

                return {
                    "original": text,
                    "translated": translated_text,
                    "confidence": 0.70,
                    "model_used": "MBart-50",
                    "translation_fallback": True,
                }
            except Exception as mbart_err:
                logger.warning(f"MBART translation fallback failed: {mbart_err}")
                return {
                    "original": text,
                    "translated": text,
                    "confidence": 0.0,
                    "model_used": "none",
                    "translation_fallback": True,
                }

    def translate_batch(
        self, texts: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for i, (text, lang) in enumerate(texts):
            if i > 0 and i % 10 == 0:
                time.sleep(0.3)
            results.append(self.translate_to_english(text, lang, is_batch_call=True))
        return results
