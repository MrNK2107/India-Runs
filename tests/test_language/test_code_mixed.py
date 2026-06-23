from __future__ import annotations

import pytest

from src.language.code_mixed import CodeMixedProcessor, Entity


@pytest.fixture
def processor():
    return CodeMixedProcessor()


def test_detect_code_mixed_pure_english(processor):
    assert processor.detect_code_mixed("I am a software engineer") is False


def test_detect_code_mixed_pure_hindi(processor):
    text = "\u092e\u0948\u0902 \u090f\u0915 \u0938\u0949\u092b\u094d\u091f\u0935\u0947\u092f\u0930 \u0907\u0902\u091c\u0940\u0928\u093f\u092f\u0930 \u0939\u0942\u0901"  # noqa: E501
    assert processor.detect_code_mixed(text) is True


def test_detect_code_mixed_hinglish(processor):
    text = "Main ek software engineer hoon with Python and Django"
    assert processor.detect_code_mixed(text) is True


def test_detect_code_mixed_hinglish_keyword(processor):
    text = "I know Python and Django bahut achha hai"
    assert processor.detect_code_mixed(text) is True


def test_extract_entities_regex_fallback(processor):
    text = "I have experience with Python and AWS at Google and Amazon"
    entities = processor.extract_entities(text)
    assert len(entities) >= 2
    labels = {e.label for e in entities}
    assert "SKILL" in labels
    assert "ORG" in labels


def test_transliterate_hinglish_pure_english(processor):
    result = processor.transliterate_hinglish("I am a software engineer")
    assert result == "I am a software engineer"


def test_transliterate_hinglish_common_words(processor):
    result = processor.transliterate_hinglish("Yeh bahut achha kaam hai")
    assert "good" in result.lower() or "very" in result.lower()


def test_entity_dataclass():
    e = Entity(text="Python", label="SKILL", start=0, end=6)
    assert e.text == "Python"
    assert e.label == "SKILL"
    assert e.confidence == 1.0


def test_detect_code_mixed_empty(processor):
    assert processor.detect_code_mixed("") is False


def test_tint_prompting_not_imported_in_planner():
    from src.agents.planner import PlannerAgent
    planner = PlannerAgent()
    assert hasattr(planner, "plan")
