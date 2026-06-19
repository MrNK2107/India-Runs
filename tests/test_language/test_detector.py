from __future__ import annotations

from src.language.detector import LanguageDetector


def test_detect_english():
    detector = LanguageDetector()
    result = detector.detect("Hello, this is a test")
    assert result["language"] == "en"


def test_detect_hindi():
    detector = LanguageDetector()
    result = detector.detect("नमस्ते, यह एक परीक्षण है")
    assert result["language"] == "hi"


def test_detect_empty():
    detector = LanguageDetector()
    result = detector.detect("")
    assert result["language"] == "en"


def test_detect_batch():
    detector = LanguageDetector()
    results = detector.detect_batch(["Hello", "नमस्ते"])
    assert len(results) == 2


def test_translator_english_passthrough():
    from src.language.translator import TranslationPipeline
    pipeline = TranslationPipeline()
    result = pipeline.translate_to_english("Hello", "en")
    assert result["translated"] == "Hello"
    assert result["confidence"] == 1.0
    assert result["translation_fallback"] is False


def test_translator_french_success(monkeypatch):
    class FakeGoogleTranslator:
        def translate(self, text: str) -> str:
            return "Hello world"

    monkeypatch.setattr("deep_translator.GoogleTranslator", lambda *a, **kw: FakeGoogleTranslator())
    from src.language.translator import TranslationPipeline
    pipeline = TranslationPipeline()
    result = pipeline.translate_to_english("Bonjour le monde", "fr")
    assert result["translation_fallback"] is False
    assert "Hello" in str(result["translated"]) or "world" in str(result["translated"])


def test_translator_batch(monkeypatch):
    class FakeGoogleTranslator:
        def translate(self, text: str) -> str:
            return "Hello"

    monkeypatch.setattr("deep_translator.GoogleTranslator", lambda *a, **kw: FakeGoogleTranslator())
    from src.language.translator import TranslationPipeline
    pipeline = TranslationPipeline()
    results = pipeline.translate_batch([("Hello", "en"), ("Hola", "es")])
    assert len(results) == 2
    assert results[0]["translation_fallback"] is False
    assert results[1]["translation_fallback"] is False
    assert "Hello" in str(results[1]["translated"])


def test_translator_supported_languages():
    from src.language.translator import SUPPORTED_LANGUAGES
    assert "hi" in SUPPORTED_LANGUAGES
    assert "ta" in SUPPORTED_LANGUAGES
    assert "fr" not in SUPPORTED_LANGUAGES


def test_multilingual_embedder_dimension():
    from src.language.multilingual import MultilingualEmbedder
    embedder = MultilingualEmbedder()
    assert embedder.dimension == 384


def test_embedder_embed_query(sample_query_text):
    from src.language.multilingual import MultilingualEmbedder
    embedder = MultilingualEmbedder()
    vec = embedder.embed_query(sample_query_text)
    assert vec.shape == (384,)
