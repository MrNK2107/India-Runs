from __future__ import annotations

import pytest

from src.matching.scorer import DEFAULT_SLIDER_WEIGHTS, CandidateScorer

SCORES = {
    "skill_match": 0.9, "experience_match": 0.1, "education_match": 0.5,
    "cross_encoder_score": 0.5, "behavioral_score": 0.5, "semantic_similarity": 0.5,
    "keyword_match": 0.5,
}


def test_slider_at_zero_excludes_the_dimension() -> None:
    # Regression: a 0 slider was treated as "not set" and fell back to the default weight
    # (experience 0.20), so 0 weighed more than e.g. 5 (0.05).
    s = CandidateScorer()
    only_skill = {k: 0.0 for k in DEFAULT_SLIDER_WEIGHTS} | {"skill_match": 100.0}
    assert s.compute_overall(SCORES, only_skill).overall == pytest.approx(0.9)
    only_exp = {k: 0.0 for k in DEFAULT_SLIDER_WEIGHTS} | {"experience_match": 100.0}
    assert s.compute_overall(SCORES, only_exp).overall == pytest.approx(0.1)


def test_slider_weight_is_monotonic() -> None:
    s = CandidateScorer()
    base = {k: v * 100 for k, v in DEFAULT_SLIDER_WEIGHTS.items()}
    # Experience scores low here, so more experience weight -> lower overall, all the way to 0.
    overall = [
        s.compute_overall(SCORES, base | {"experience_match": w}).overall
        for w in (40.0, 20.0, 5.0, 0.0)
    ]
    assert overall == sorted(overall)


def test_missing_slider_keys_use_defaults() -> None:
    s = CandidateScorer()
    partial = s.compute_overall(SCORES, {"skill_match": 25.0})
    full = s.compute_overall(SCORES, {k: v * 100 for k, v in DEFAULT_SLIDER_WEIGHTS.items()})
    assert partial.overall == pytest.approx(full.overall)


def test_all_sliders_zero_gives_zero_not_error() -> None:
    s = CandidateScorer()
    assert s.compute_overall(SCORES, {k: 0.0 for k in DEFAULT_SLIDER_WEIGHTS}).overall == 0.0


def test_no_sliders_uses_internal_weights() -> None:
    s = CandidateScorer()
    assert 0.0 <= s.compute_overall(SCORES, None).overall <= 1.0
    assert s.compute_overall(SCORES, {}).overall == s.compute_overall(SCORES, None).overall
