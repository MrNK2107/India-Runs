from __future__ import annotations

import pytest

from src.core.models import (
    Education,
    Location,
    MatchMetadata,
    MatchResult,
    MatchScores,
    PersonalInfo,
    Profile,
    ProfileMetadata,
    ProfileSource,
    SearchMethod,
    WorkExperience,
)
from src.fairness.bias_detector import BiasDetector
from src.fairness.groups import city_key, is_top_institution
from src.fairness.metrics import _get_city_tier, _is_top_university


def _profile(pid: str, name: str = "Asha", city: str | None = None, inst: str | None = None,
             lang: str = "en", exp_location: str | None = None) -> Profile:
    return Profile(
        profile_id=pid,
        source=ProfileSource.MANUAL,
        personal=PersonalInfo(name=name, location=Location(city=city) if city else Location()),
        education=[Education(institution=inst)] if inst else [],
        experience=(
            [WorkExperience(title="SDE", company="Acme", location=exp_location)]
            if exp_location else []
        ),
        metadata=ProfileMetadata(language_detected=lang),
    )


def _match(pid: str, score: float) -> MatchResult:
    return MatchResult(
        query_id="q", profile_id=pid, rank=1,
        scores=MatchScores(overall=score, semantic_similarity=0, keyword_match=0,
                           skill_match=0, experience_match=0, confidence=0.5),
        metadata=MatchMetadata(search_method=SearchMethod.HYBRID),
    )


# ------------------------------------------------------------ group helpers


@pytest.mark.parametrize("inst", [
    "IIT Bombay", "Indian Institute of Technology, Madras", "NIT Trichy", "BITS Pilani",
    "IIIT Hyderabad", "National Institute of Technology Karnataka", "iit-delhi",
    "Birla Institute of Technology and Science",
])
def test_top_institutions(inst: str) -> None:
    assert is_top_institution(inst)


@pytest.mark.parametrize("inst", [
    "KIIT University", "Community College of Pune", "Infinity Institute", "Unitedworld School",
    "Amity University", "Habits Academy", "", None,
])
def test_not_top_institutions(inst: str | None) -> None:
    # Substring matching counted KIIT ("iit"), Community/Infinity/United ("nit") as IIT/NIT.
    assert not is_top_institution(inst)


def test_city_key() -> None:
    assert city_key("Bengaluru, Karnataka") == "bengaluru"
    assert city_key("  New Delhi ") == "new delhi"
    assert city_key(None) == ""


def test_fairness_metrics_use_shared_helpers() -> None:
    assert not _is_top_university(_profile("p", inst="KIIT University"))
    assert _is_top_university(_profile("p", inst="IIT Madras"))
    # Experience locations carry the state ("Bengaluru, Karnataka"): was classed tier-2.
    assert _get_city_tier(_profile("p", exp_location="Bengaluru, Karnataka")) == "tier1"


# ---------------------------------------------------------- group bias checks


def test_one_empty_group_is_not_reported_as_bias() -> None:
    # Regression: an empty group averaged to 0, so "all candidates are tier-1" gave
    # diff = avg score -> detected = True.
    profiles = {f"p{i}": _profile(f"p{i}", city="Mumbai") for i in range(5)}
    matches = [_match(f"p{i}", 0.8) for i in range(5)]
    d = BiasDetector()
    for kind in ("location", "university", "language"):
        res = d.detect_bias(matches, profiles, kind)
        assert res["detected"] is False, kind
        assert any("insufficient" in o for o in res["observations"]), kind


def test_location_bias_detected_and_experience_location_used() -> None:
    profiles = {
        "a": _profile("a", city="Bangalore"),
        "b": _profile("b", exp_location="Hyderabad, Telangana"),
        "c": _profile("c", city="Indore"),
        "d": _profile("d", city="Bhopal"),
    }
    matches = [_match("a", 0.9), _match("b", 0.9), _match("c", 0.5), _match("d", 0.5)]
    res = BiasDetector().check_location_bias(matches, profiles)
    assert res["detected"] is True
    assert res["details"]["tier1_avg"] == pytest.approx(0.9)
    assert res["details"]["tier2_avg"] == pytest.approx(0.5)


def test_university_bias_not_fooled_by_kiit() -> None:
    profiles = {
        "a": _profile("a", inst="IIT Delhi"),
        "b": _profile("b", inst="KIIT University"),
    }
    matches = [_match("a", 0.6), _match("b", 0.9)]
    res = BiasDetector().check_university_bias(matches, profiles)
    assert res["details"]["top_univ_avg"] == pytest.approx(0.6)
    assert res["details"]["other_avg"] == pytest.approx(0.9)
    assert res["detected"] is False


def test_language_bias_detected() -> None:
    profiles = {"a": _profile("a"), "b": _profile("b", lang="hi")}
    res = BiasDetector().check_language_bias([_match("a", 0.9), _match("b", 0.5)], profiles)
    assert res["detected"] is True


def test_name_bias_can_be_detected() -> None:
    # Previously `detected` could never be True (no observation was ever tagged).
    profiles = {}
    matches = []
    for i in range(3):
        profiles[f"a{i}"] = _profile(f"a{i}", name=f"Arun {i}")
        matches.append(_match(f"a{i}", 0.9))
        profiles[f"z{i}"] = _profile(f"z{i}", name=f"Zoya {i}")
        matches.append(_match(f"z{i}", 0.4))
    res = BiasDetector().check_name_bias(matches, profiles)
    assert res["detected"] is True
    assert any(o.endswith("(potential bias)") for o in res["observations"])


def test_name_bias_balanced_is_not_detected() -> None:
    profiles = {f"p{i}": _profile(f"p{i}", name=n) for i, n in enumerate("AAABBB")}
    matches = [_match(f"p{i}", 0.7) for i in range(6)]
    assert BiasDetector().check_name_bias(matches, profiles)["detected"] is False


def test_unknown_type_and_missing_profiles() -> None:
    d = BiasDetector()
    assert d.detect_bias([], {}, "zodiac")["detected"] is False
    res = d.detect_bias([_match("ghost", 0.9)], {}, "location")
    assert res["detected"] is False
