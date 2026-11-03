from __future__ import annotations

import pytest

from src.language.multilingual import DEFAULT_EMBEDDING_MODEL, MultilingualEmbedder


def test_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert MultilingualEmbedder().model_name == DEFAULT_EMBEDDING_MODEL


def test_env_override_and_explicit_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    assert MultilingualEmbedder().model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert MultilingualEmbedder("explicit").model_name == "explicit"
