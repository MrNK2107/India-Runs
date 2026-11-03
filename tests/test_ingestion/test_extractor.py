"""Tests for the LLM-backed FieldExtractor. The LLM client is always faked (no network)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.ingestion.extractor import FieldExtractor


class FakeClient:
    """Mimics a LangChain chat model: `await ainvoke(prompt)` -> object with `.content`."""

    def __init__(self, content: Any = None, exc: Exception | None = None) -> None:
        self.content = content
        self.exc = exc
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(content=self.content)


def make(content: Any = None, exc: Exception | None = None) -> tuple[FieldExtractor, FakeClient]:
    ex = FieldExtractor()
    client = FakeClient(content, exc)
    ex._client = client
    return ex, client


SKILLS = [
    {"name": "Python", "category": "programming_language", "proficiency": "advanced"},
    {"name": "AWS", "category": "cloud", "proficiency": "intermediate"},
]
EXPERIENCE = [
    {"title": "SDE", "company": "Acme", "start_date": "2020-01", "end_date": None,
     "description": "Built APIs"},
]
EDUCATION = [
    {"institution": "IIT Madras", "degree": "B.Tech", "field": "CSE",
     "start_year": 2015, "end_year": 2019},
]


# ---------------------------------------------------------------- happy paths


async def test_extract_skills_plain_json() -> None:
    ex, client = make(json.dumps(SKILLS))
    assert await ex.extract_skills("Python dev") == SKILLS
    assert len(client.prompts) == 1


async def test_extract_experience_and_education() -> None:
    ex, _ = make(json.dumps(EXPERIENCE))
    assert await ex.extract_experience("x") == EXPERIENCE
    ex, _ = make(json.dumps(EDUCATION))
    assert await ex.extract_education("x") == EDUCATION


async def test_extract_all_returns_all_three_keys() -> None:
    payload = {"skills": SKILLS, "experience": EXPERIENCE, "education": EDUCATION}
    ex, _ = make(json.dumps(payload))
    assert await ex.extract_all("x") == payload


@pytest.mark.parametrize(
    "wrapped",
    [
        "```json\n{body}\n```",
        "```\n{body}\n```",
        "```JSON\n{body}\n```",
        "  ```json {body} ```  ",
        "Here is the JSON:\n```json\n{body}\n```\nHope this helps!",
        "Sure! {body}",
    ],
)
async def test_fenced_or_chatty_responses_are_parsed(wrapped: str) -> None:
    ex, _ = make(wrapped.replace("{body}", json.dumps(SKILLS)))
    assert await ex.extract_skills("x") == SKILLS


# --------------------------------------------------------------- prompt shape


def test_prompt_includes_schema_and_truncates_text() -> None:
    ex = FieldExtractor()
    text = "A" * 5000
    prompt = ex._build_extraction_prompt(text, "skills")
    assert "Return ONLY valid JSON" in prompt
    assert '"proficiency"' in prompt
    assert "A" * 4000 in prompt and "A" * 4001 not in prompt


def test_unknown_target_falls_back_to_all_schema() -> None:
    prompt = FieldExtractor()._build_extraction_prompt("t", "bogus")
    assert '"skills": [...]' in prompt


def test_prompt_keeps_unicode_text() -> None:
    prompt = FieldExtractor()._build_extraction_prompt("Anaïs Ñúñez — 数据 科学家", "all")
    assert "Anaïs Ñúñez — 数据 科学家" in prompt


# ------------------------------------------------------ malformed / partial


@pytest.mark.parametrize("bad", ["", "   ", "not json", "{\"skills\": [", "```json\n```", "null"])
async def test_malformed_list_targets_return_empty_list(bad: str) -> None:
    ex, _ = make(bad)
    assert await ex.extract_skills("x") == []
    assert await ex.extract_experience("x") == []
    assert await ex.extract_education("x") == []


@pytest.mark.parametrize("bad", ["", "garbage", "[1, 2", "null", "42", "[]"])
async def test_malformed_all_returns_empty_sections(bad: str) -> None:
    ex, _ = make(bad)
    assert await ex.extract_all("x") == {"skills": [], "experience": [], "education": []}


async def test_llm_exception_returns_typed_empty_result() -> None:
    # _call_llm used to return "{}" on error, which parsed to a dict for list targets.
    ex, _ = make(exc=RuntimeError("timeout"))
    assert await ex.extract_skills("x") == []
    assert await ex.extract_all("x") == {"skills": [], "experience": [], "education": []}


async def test_list_target_unwraps_object_keyed_by_target() -> None:
    ex, _ = make(json.dumps({"skills": SKILLS}))
    assert await ex.extract_skills("x") == SKILLS


async def test_list_target_given_unrelated_object_returns_empty() -> None:
    ex, _ = make(json.dumps({"foo": "bar"}))
    assert await ex.extract_skills("x") == []


async def test_list_target_given_single_object_wraps_it() -> None:
    ex, _ = make(json.dumps(EDUCATION[0]))
    assert await ex.extract_education("x") == EDUCATION


async def test_non_dict_items_are_dropped() -> None:
    ex, _ = make(json.dumps([SKILLS[0], "Python", 3, None, [1], SKILLS[1]]))
    assert await ex.extract_skills("x") == SKILLS


async def test_all_fills_missing_and_mistyped_sections() -> None:
    ex, _ = make(json.dumps({"skills": SKILLS, "experience": "n/a", "extra": 1}))
    assert await ex.extract_all("x") == {
        "skills": SKILLS, "experience": [], "education": [], "extra": 1,
    }


async def test_content_none_or_block_list_is_handled() -> None:
    ex, _ = make(None)
    assert await ex.extract_skills("x") == []
    # Some chat models return content as a list of blocks.
    ex, _ = make([{"type": "text", "text": json.dumps(SKILLS)}])
    assert await ex.extract_skills("x") == SKILLS


# ------------------------------------------------------------------ encoding


async def test_unicode_and_escaped_values_roundtrip() -> None:
    skills = [{"name": "Ruby on Rails ✓", "category": "framework", "proficiency": "expert"}]
    ex, _ = make(json.dumps(skills, ensure_ascii=False))
    assert await ex.extract_skills("x") == skills
    ex, _ = make(json.dumps(skills, ensure_ascii=True))
    assert await ex.extract_skills("x") == skills


async def test_bom_prefixed_response_is_parsed() -> None:
    ex, _ = make("﻿" + json.dumps(SKILLS))
    assert await ex.extract_skills("x") == SKILLS


# ---------------------------------------------------------------- duplicates


async def test_duplicate_skills_are_deduped_case_insensitively() -> None:
    dupes = [
        {"name": "Python", "proficiency": "advanced"},
        {"name": " python ", "proficiency": "beginner"},
        {"name": "PYTHON"},
        {"name": "AWS"},
    ]
    ex, _ = make(json.dumps(dupes))
    out = await ex.extract_skills("x")
    assert [s["name"] for s in out] == ["Python", "AWS"]
    assert out[0]["proficiency"] == "advanced"  # first occurrence wins


async def test_exact_duplicate_experience_and_education_rows_are_dropped() -> None:
    ex, _ = make(json.dumps(EXPERIENCE + EXPERIENCE))
    assert await ex.extract_experience("x") == EXPERIENCE
    ex, _ = make(json.dumps({"skills": SKILLS + SKILLS, "experience": [],
                             "education": EDUCATION + EDUCATION}))
    out = await ex.extract_all("x")
    assert out["skills"] == SKILLS and out["education"] == EDUCATION


async def test_distinct_roles_at_same_company_are_kept() -> None:
    rows = [dict(EXPERIENCE[0]), dict(EXPERIENCE[0], title="Senior SDE")]
    ex, _ = make(json.dumps(rows))
    assert await ex.extract_experience("x") == rows


async def test_skills_without_a_usable_name_are_dropped() -> None:
    rows = [{"category": "x"}, {"name": None}, {"name": "  "}, {"name": 5}, {"name": "Go"}]
    ex, _ = make(json.dumps(rows))
    assert await ex.extract_skills("x") == [{"name": "Go"}]


# ----------------------------------------------------------------- client init


def test_client_is_created_lazily_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_get() -> FakeClient:
        calls.append(1)
        return FakeClient("[]")

    monkeypatch.setattr("src.core.config.get_llm_client", fake_get)
    ex = FieldExtractor()
    assert ex._client is None
    c1 = ex._get_client()
    c2 = ex._get_client()
    assert c1 is c2 and len(calls) == 1


async def test_content_string_blocks_and_non_string_content() -> None:
    ex, _ = make(["[", {"type": "image"}, json.dumps(SKILLS)[1:]])
    assert await ex.extract_skills("x") == SKILLS
    ex, _ = make(12345)  # unexpected type: stringified, then fails to parse cleanly
    assert await ex.extract_skills("x") == []


async def test_fence_regex_does_not_drop_json_containing_backticks_outside() -> None:
    body = json.dumps([{"name": "C++"}])
    ex, _ = make(f"Use `json`:\n```json\n{body}\n```")
    assert await ex.extract_skills("x") == [{"name": "C++"}]
