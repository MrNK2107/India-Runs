from __future__ import annotations

import pytest

from src.core import config


def test_load_yaml_config_reads_utf8_regardless_of_locale(tmp_path, monkeypatch) -> None:
    # "ā" encodes to C4 81; 0x81 is undefined in cp1252 (the Windows default here), so a
    # locale-dependent open() raised UnicodeDecodeError. configs/*.yaml contain em dashes.
    (tmp_path / "x.yaml").write_text("# Bengalūru — Āā\ncity: Bengalūru Āā\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIGS_DIR", tmp_path)
    assert config.load_yaml_config("x.yaml") == {"city": "Bengalūru Āā"}


def test_load_yaml_config_missing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "CONFIGS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        config.load_yaml_config("nope.yaml")


def test_load_yaml_config_empty_file_returns_empty_dict(tmp_path, monkeypatch) -> None:
    (tmp_path / "empty.yaml").write_text("# only a comment\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIGS_DIR", tmp_path)
    assert config.load_yaml_config("empty.yaml") == {}


def test_repo_configs_load() -> None:
    scoring = config.load_yaml_config("scoring_weights.yaml")
    assert scoring["rrf_k"] == 60
    assert "min_good_matches_for_pass" in scoring
    assert isinstance(config.load_yaml_config("settings.yaml"), dict)


def test_load_yaml_config_rejects_non_mapping(tmp_path, monkeypatch) -> None:
    (tmp_path / "list.yaml").write_text("- a\n- b\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIGS_DIR", tmp_path)
    with pytest.raises(ValueError, match="must contain a mapping"):
        config.load_yaml_config("list.yaml")
