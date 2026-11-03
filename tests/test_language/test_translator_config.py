from __future__ import annotations

from src.language import translator


def test_mbart_model_name_read_from_settings_models_translation(monkeypatch) -> None:
    # settings.yaml nests it under models.translation.fallback; the code read translation.fallback,
    # so the configured value was silently ignored.
    cfg = {"models": {"translation": {"fallback": "my-org/custom-mbart"}}}
    monkeypatch.setattr("src.core.config.get_app_config", lambda: cfg)
    assert translator.mbart_model_name() == "my-org/custom-mbart"


def test_mbart_model_name_defaults_when_missing_or_broken(monkeypatch) -> None:
    monkeypatch.setattr("src.core.config.get_app_config", lambda: {})
    assert translator.mbart_model_name() == translator.DEFAULT_MBART_MODEL

    def boom() -> dict:
        raise FileNotFoundError("settings.yaml")

    monkeypatch.setattr("src.core.config.get_app_config", boom)
    assert translator.mbart_model_name() == translator.DEFAULT_MBART_MODEL


def test_repo_settings_yaml_value() -> None:
    assert translator.mbart_model_name() == "facebook/mbart-large-50-many-to-many-mmt"


def test_mbart_fallback_loads_configured_model(monkeypatch) -> None:
    """Google fails -> MBART path loads the model named in settings.yaml (no real download)."""
    import sys
    import types

    loaded: list[str] = []

    class FakeTok:
        lang_code_to_id = {"en_XX": 2}
        src_lang = ""

        @classmethod
        def from_pretrained(cls, name: str) -> FakeTok:
            loaded.append(name)
            return cls()

        def __call__(self, text: str, return_tensors: str) -> dict:
            return {"input_ids": [1]}

        def batch_decode(self, tokens, skip_special_tokens: bool) -> list[str]:
            return ["hello world"]

    class FakeModel:
        @classmethod
        def from_pretrained(cls, name: str) -> FakeModel:
            loaded.append(name)
            return cls()

        def generate(self, **kw) -> list:
            return [[2, 3]]

    fake_tf = types.SimpleNamespace(
        MBart50TokenizerFast=FakeTok, MBartForConditionalGeneration=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setattr(
        "src.core.config.get_app_config",
        lambda: {"models": {"translation": {"fallback": "my-org/custom-mbart"}}},
    )

    class Broken:
        def translate(self, text: str) -> str:
            raise ConnectionError("offline")

    pipeline = translator.TranslationPipeline()
    pipeline._translator = Broken()  # type: ignore[assignment]
    out = pipeline.translate_to_english("namaste duniya", "hi", is_batch_call=True)
    assert out["model_used"] == "MBart-50" and out["translated"] == "hello world"
    assert loaded == ["my-org/custom-mbart", "my-org/custom-mbart"]
