from __future__ import annotations

import statistics
from datetime import UTC, datetime, timedelta

import pytest

from src.core.models import (
    PersonalInfo,
    ProfessionalInfo,
    ProficiencyLevel,
    Profile,
    ProfileSource,
    Signals,
    Skill,
    WorkExperience,
)
from src.matching.behavioral_scorer import (
    compute_behavioral_score,
    compute_career_trajectory,
    compute_skill_proficiency,
    detect_honeypot,
)


def _profile(
    experience: list[WorkExperience] | None = None,
    skills: list[Skill] | None = None,
    years: float | None = None,
    signals: Signals | None = None,
) -> Profile:
    return Profile(
        profile_id="p",
        source=ProfileSource.MANUAL,
        personal=PersonalInfo(name="X"),
        professional=ProfessionalInfo(total_experience_years=years),
        experience=experience or [],
        skills=skills or [],
        signals=signals or Signals(),
    )


def _job(title: str, company: str, start: str | None, end: str | None = None) -> WorkExperience:
    return WorkExperience(title=title, company=company, start_date=start, end_date=end)


def _days_ago(n: int) -> str:
    return (datetime.now(UTC) - timedelta(days=n)).strftime("%Y-%m-%d")


# ── detect_honeypot ─────────────────────────────────────────────────


def test_honeypot_clean_profile() -> None:
    p = _profile(
        experience=[_job("Engineer", "Infosys", "2015-06", "2019-01"),
                    _job("Senior Engineer", "Swiggy", "2019-02", "2023-01")],
        skills=[Skill(name="Python", proficiency=ProficiencyLevel.EXPERT, years_used=6)],
        years=8,
    )
    assert detect_honeypot(p) is None


def test_honeypot_time_travel() -> None:
    p = _profile(experience=[_job("Dev", "  Swiggy ", "2010-01-01", "2012-01-01")])
    reason = detect_honeypot(p)
    assert reason is not None and "before" in reason and "2014" in reason


def test_honeypot_ignores_unknown_company_and_bad_dates() -> None:
    p = _profile(experience=[_job("Dev", "Unknown Ltd", "1900-01"),
                             _job("Dev", "Zepto", "garbage")])
    assert detect_honeypot(p) is None


def test_honeypot_skill_density() -> None:
    skills = [Skill(name=f"s{i}") for i in range(11)]
    reason = detect_honeypot(_profile(skills=skills, years=2))
    assert reason is not None and "11 skills in 2y" in reason
    assert detect_honeypot(_profile(skills=skills[:10], years=2)) is None  # exactly 5/yr is ok
    assert detect_honeypot(_profile(skills=skills, years=0)) is None  # no div-by-zero


def test_honeypot_expert_zero_years() -> None:
    experts = [Skill(name=f"s{i}", proficiency=ProficiencyLevel.EXPERT, years_used=0)
               for i in range(5)]
    reason = detect_honeypot(_profile(skills=experts))
    assert reason is not None and "5 expert skills" in reason
    assert detect_honeypot(_profile(skills=experts[:4])) is None


def test_honeypot_long_gap_and_multiple_reasons() -> None:
    p = _profile(experience=[
        _job("Dev", "Acme", "2005-01", "2008-01"),
        _job("Dev", "Swiggy", "2010-01", "2011-01"),  # time travel too
        _job("Dev", "Beta", "2017-03", "2019-01"),  # 74-month gap
    ])
    reason = detect_honeypot(p)
    assert reason is not None
    assert "Gap of 6y" in reason
    assert "; " in reason  # several reasons are joined


def test_honeypot_gap_of_exactly_five_years_is_ok() -> None:
    p = _profile(experience=[_job("Dev", "A", "2005-01", "2010-01"),
                             _job("Dev", "B", "2015-01", "2016-01")])
    assert detect_honeypot(p) is None


# ── compute_career_trajectory ──────────────────────────────────────


def test_trajectory_no_experience_is_neutral() -> None:
    assert compute_career_trajectory(_profile()) == 0.3


def test_trajectory_job_hopper_scores_below_stable() -> None:
    hopper = _profile(experience=[
        _job("Engineer", "A", "2020-01", "2020-10"),
        _job("Engineer", "B", "2020-11", "2021-08"),
        _job("Engineer", "C", "2021-09", "2022-06"),
    ])
    stable = _profile(experience=[
        _job("Engineer", "A", "2010-01", "2014-01"),
        _job("Engineer", "B", "2014-02", "2018-02"),
        _job("Engineer", "C", "2018-03", "2022-03"),
    ])
    # hopper: tenure 0.3, not consulting 0.85, no progression 0.3
    assert compute_career_trajectory(hopper) == pytest.approx(statistics.mean([0.3, 0.85, 0.3]))
    assert compute_career_trajectory(stable) == pytest.approx(statistics.mean([0.9, 0.85, 0.3]))


def test_trajectory_progression_rewards_promotions() -> None:
    # experience is ordered most-recent first
    promoted = _profile(experience=[
        _job("Senior Engineer", "A", "2019-01", "2023-01"),
        _job("Engineer", "A", "2015-01", "2019-01"),
    ])
    score = compute_career_trajectory(promoted)
    assert score == pytest.approx(statistics.mean([0.8, 0.85, 0.9]))


def test_trajectory_single_job_without_dates() -> None:
    p = _profile(experience=[_job("Engineer", "A", None)])
    assert compute_career_trajectory(p) == pytest.approx(statistics.mean([0.5, 0.85, 0.6]))


def test_trajectory_full_consulting_career_penalised() -> None:
    p = _profile(experience=[_job("Freelance Developer", "Self", None),
                             _job("Engineer", "XYZ Consulting", None)])
    assert compute_career_trajectory(p) == pytest.approx(statistics.mean([0.5, 0.25, 0.3]))


def test_trajectory_staffing_firm_company_counts_as_consulting() -> None:
    # Regression: only the first three keywords were checked against the
    # company name, so staffing firms listed by name (Randstad, Adecco,
    # TeamLease, ...) were never detected unless the *title* mentioned them.
    p = _profile(experience=[_job("Software Engineer", "Randstad India", None),
                             _job("Software Engineer", "TeamLease Services", None)])
    assert compute_career_trajectory(p) == pytest.approx(statistics.mean([0.5, 0.25, 0.3]))


# ── compute_behavioral_score ───────────────────────────────────────


def test_behavioral_all_unknown_is_low_neutral() -> None:
    expected = statistics.mean([0.3, 0.3, 0.2, 0.3, 0.2, 0.4, 0.4, 0.5, 0.2])
    assert compute_behavioral_score(Signals()) == pytest.approx(expected)


def test_behavioral_strong_candidate_beats_weak() -> None:
    strong = Signals(
        recruiter_response_rate=0.9, saved_by_recruiters_30d=20,
        profile_completeness_score=95, verified_email=True, verified_phone=True,
        linkedin_connected=True, github_activity_score=40, open_to_work=True,
        willing_to_relocate=True, interview_completion_rate=0.95,
        last_active_date=_days_ago(1),
    )
    weak = Signals(
        recruiter_response_rate=0.05, saved_by_recruiters_30d=0,
        profile_completeness_score=20, verified_email=False, verified_phone=False,
        linkedin_connected=False, github_activity_score=0, open_to_work=False,
        willing_to_relocate=False, interview_completion_rate=0.1,
        last_active_date=_days_ago(400),
    )
    s, w = compute_behavioral_score(strong), compute_behavioral_score(weak)
    assert 0.9 < s <= 1.0
    assert 0.0 <= w < 0.3
    assert s > w


def test_behavioral_verification_is_fraction_of_known() -> None:
    half = compute_behavioral_score(Signals(verified_email=True, verified_phone=False))
    base = compute_behavioral_score(Signals())
    # Only the verification component changes: 0.3 (unknown) -> 0.5.
    assert half - base == pytest.approx((0.5 - 0.3) / 9)


def test_behavioral_bad_last_active_date_falls_back() -> None:
    bad = compute_behavioral_score(Signals(last_active_date="20/05/2026"))
    base = compute_behavioral_score(Signals())
    assert bad - base == pytest.approx((0.3 - 0.2) / 9)


def test_behavioral_negative_rates_treated_as_unknown() -> None:
    s = Signals(recruiter_response_rate=-1, interview_completion_rate=-1,
                github_activity_score=-5)
    base = compute_behavioral_score(Signals())
    # github -5 clamps to 0 (vs 0.2 unknown); the two rates fall back to defaults
    assert compute_behavioral_score(s) - base == pytest.approx((0.0 - 0.2) / 9)


@pytest.mark.parametrize(
    "signals",
    [
        Signals(profile_completeness_score=250),
        Signals(profile_completeness_score=-40),
        Signals(interview_completion_rate=87.0),  # percent instead of fraction
        Signals(saved_by_recruiters_30d=-3),
        Signals(last_active_date=(datetime.now(UTC) + timedelta(days=30)).strftime("%Y-%m-%d")),
    ],
)
def test_behavioral_score_always_in_unit_interval(signals: Signals) -> None:
    # Regression: out-of-range raw signals leaked straight through (e.g.
    # completeness 250 -> component 2.5), so the mean could leave [0, 1].
    worst = Signals(
        profile_completeness_score=signals.profile_completeness_score,
        interview_completion_rate=signals.interview_completion_rate,
        saved_by_recruiters_30d=signals.saved_by_recruiters_30d,
        last_active_date=signals.last_active_date,
        recruiter_response_rate=1.0, github_activity_score=999, open_to_work=True,
        willing_to_relocate=True, verified_email=True,
    )
    for s in (signals, worst):
        assert 0.0 <= compute_behavioral_score(s) <= 1.0


def test_behavioral_out_of_range_inputs_are_clamped() -> None:
    hi = Signals(profile_completeness_score=250)
    exact = Signals(profile_completeness_score=100)
    assert compute_behavioral_score(hi) == pytest.approx(compute_behavioral_score(exact))
    pct = Signals(interview_completion_rate=87.0)
    full = Signals(interview_completion_rate=1.0)
    assert compute_behavioral_score(pct) == pytest.approx(compute_behavioral_score(full))


# ── compute_skill_proficiency ──────────────────────────────────────


def test_skill_proficiency_empty_inputs() -> None:
    p = _profile(skills=[Skill(name="Python")])
    assert compute_skill_proficiency(p, []) == 0.0
    assert compute_skill_proficiency(_profile(), ["python"]) == 0.0
    assert compute_skill_proficiency(p, ["rust"]) == 0.0


def test_skill_proficiency_levels_and_bonuses() -> None:
    def score(**kw: object) -> float:
        p = _profile(skills=[Skill(name="Python", confidence=0.0, **kw)])  # type: ignore[arg-type]
        return compute_skill_proficiency(p, ["PYTHON"])

    assert score() == pytest.approx(0.5)
    assert score(proficiency=ProficiencyLevel.BEGINNER) == pytest.approx(0.5)
    assert score(proficiency=ProficiencyLevel.INTERMEDIATE) == pytest.approx(0.7)
    assert score(proficiency=ProficiencyLevel.ADVANCED) == pytest.approx(0.85)
    assert score(proficiency=ProficiencyLevel.EXPERT) == pytest.approx(1.0)
    assert score(proficiency=ProficiencyLevel.INTERMEDIATE, years_used=5) == pytest.approx(0.85)
    assert score(proficiency=ProficiencyLevel.ADVANCED, years_used=20) == pytest.approx(1.0)


def test_skill_proficiency_confidence_bonus() -> None:
    p = _profile(skills=[Skill(name="Go", confidence=1.0)])
    assert compute_skill_proficiency(p, ["go"]) == pytest.approx(0.6)


def test_skill_proficiency_assessment_overrides_when_higher() -> None:
    sig = Signals(skill_assessment_scores={"NLP": 92.0, "SQL": 30.0})
    p = _profile(
        skills=[Skill(name="NLP", confidence=0.0), Skill(name="SQL", confidence=0.0),
                Skill(name="Excel", confidence=0.0)],
        signals=sig,
    )
    # NLP -> 0.92 (test beats base 0.5), SQL stays 0.5 (lower test ignored)
    assert compute_skill_proficiency(p, ["nlp", "sql"]) == pytest.approx((0.92 + 0.5) / 2)
