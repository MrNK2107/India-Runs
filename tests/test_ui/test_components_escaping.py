"""Profile data reaches gr.HTML (which does not sanitise) -> it must be HTML-escaped.

Profiles can now come from arbitrary POST /ingest uploads, so a name like
`<img src=x onerror=...>` would run script in the recruiter's browser.
"""
from __future__ import annotations

import math

from src.core.models import (
    MatchRecommendation,
    MatchScores,
    Rationale,
    SearchResultItem,
    SkillDetail,
)
from src.ui.components import (
    _score_bar,
    create_candidate_card,
    create_error_panel,
    create_rationale_panel,
    create_skill_match_table,
)

XSS = '<img src=x onerror="alert(1)">'


def _assert_escaped(html: str) -> None:
    assert "<img" not in html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html


def _item(**kw: object) -> SearchResultItem:
    base: dict[str, object] = dict(
        rank=1, profile_id="P1", name="Asha", scores=MatchScores(overall=0.8),
        rationale=Rationale(summary="ok"),
    )
    base.update(kw)
    return SearchResultItem(**base)  # type: ignore[arg-type]


def test_candidate_card_escapes_every_text_field() -> None:
    for field in ("name", "profile_id", "current_title", "current_company", "location"):
        _assert_escaped(create_candidate_card(_item(**{field: XSS})))
    _assert_escaped(create_candidate_card(_item(matched_skills=[XSS])))
    _assert_escaped(create_candidate_card(_item(missing_skills=[XSS])))


def test_candidate_card_keeps_plain_unicode() -> None:
    html = create_candidate_card(_item(name="Anaïs Ñúñez", current_company="Tata & Sons"))
    assert "Anaïs Ñúñez" in html
    assert "Tata &amp; Sons" in html


def test_rationale_panel_escapes() -> None:
    r = Rationale(summary=XSS, strengths=[XSS], gaps=[XSS],
                  recommendation=MatchRecommendation.GOOD)
    html = create_rationale_panel(r, XSS)
    assert "<img" not in html
    assert html.count("&lt;img") == 4


def test_skill_table_escapes() -> None:
    detail = SkillDetail(skill=XSS, required=True, found=True, proficiency_match=True,
                         evidence=XSS)
    r = Rationale(summary="s", skill_details=[detail])
    html = create_skill_match_table(r)
    assert "<img" not in html and html.count("&lt;img") == 2


def test_error_panel_escapes_exception_text() -> None:
    _assert_escaped(create_error_panel(f"Search failed: {XSS}"))


def test_score_bar_survives_nan_and_out_of_range() -> None:
    assert "0%" in _score_bar("Skill", math.nan)
    assert "100%" in _score_bar("Skill", 7.0)
    assert "0%" in _score_bar("Skill", -1.0)
