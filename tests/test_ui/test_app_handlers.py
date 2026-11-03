"""Regression tests for src/ui/app.py handler return shapes (found via mypy)."""

import asyncio
import json

import pytest

app = pytest.importorskip("src.ui.app")

# parse_query_to_ui is wired to 5 Gradio outputs; every path must return 5 values.
N_OUTPUTS = 5


@pytest.mark.parametrize("query", ["", "   "])
def test_parse_query_to_ui_empty_query_returns_five_outputs(query: str) -> None:
    out = asyncio.run(app.parse_query_to_ui(query, True))
    assert len(out) == N_OUTPUTS
    assert out[-1] is True


def test_parse_query_to_ui_json_query_returns_five_outputs() -> None:
    q = json.dumps({"required_skills": ["python"], "original_query": "python dev"})
    out = asyncio.run(app.parse_query_to_ui(q, False))
    assert len(out) == N_OUTPUTS
    assert out[-1] is False


def test_search_handler_without_orchestrator_returns_error_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.api.routes.search as search_routes

    monkeypatch.setattr(app, "_ensure_search_system", lambda: True)
    monkeypatch.setattr(search_routes, "_orchestrator", None)
    out = asyncio.run(
        app.search_handler(
            "python dev", "", 0, False, 10, True, None, progress=lambda *a, **k: None
        )
    )
    assert len(out) == 4
    assert out[2] == "[]"


def test_re_rank_handler_reorders_by_new_weights_and_escapes() -> None:
    def row(pid: str, name: str, skill: float, exp: float) -> dict:
        return {
            "rank": 1, "profile_id": pid, "name": name,
            "scores": {"overall": 0.5, "skill_match": skill, "experience_match": exp,
                       "semantic_similarity": 0.5, "keyword_match": 0.5, "confidence": 0.5},
            "matched_skills": [], "missing_skills": [], "rationale": {},
        }

    cached = json.dumps([
        row("A", "Skilled <b>", skill=0.9, exp=0.1),
        row("B", "Experienced", skill=0.1, exp=0.9),
    ])
    # Sliders in SLIDER_KEYS order; weight skills only, then experience only.
    keys = list(app.SLIDER_KEYS)
    skills_only = [100.0 if k == "skill_match" else 0.0 for k in keys]
    exp_only = [100.0 if k == "experience_match" else 0.0 for k in keys]
    out = app.re_rank_handler(cached, *skills_only)
    assert isinstance(out, str)
    assert out.index("Skilled &lt;b&gt;") < out.index("Experienced")
    out = app.re_rank_handler(cached, *exp_only)
    assert out.index("Experienced") < out.index("Skilled &lt;b&gt;")


def test_re_rank_handler_bad_cache() -> None:
    assert "No results" in app.re_rank_handler("[]")
    assert "Could not parse" in app.re_rank_handler("{not json")
