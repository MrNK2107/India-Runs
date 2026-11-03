from __future__ import annotations

from src.core.models import (
    Location,
    PersonalInfo,
    ProfessionalInfo,
    ProficiencyLevel,
    Profile,
    Signals,
    Skill,
    WorkExperience,
)
from src.core.query_parser import parse_query
from src.explain.evidence import explain_candidate, find_evidence, trajectory_facts

WEIGHTS = {
    "skill_match": 0.2, "semantic_similarity": 0.2, "cross_encoder_score": 0.15,
    "behavioral_score": 0.15, "keyword_match": 0.1, "experience_match": 0.08,
    "career_trajectory_score": 0.07, "skill_proficiency_score": 0.05,
}


def _profile(**overrides: object) -> Profile:
    base: dict[str, object] = dict(
        profile_id="P1",
        raw_text=(
            "Name: Asha. Title: Data Engineer. Summary: Data person. Toolkit: Python, SQL, "
            "Spark, Kubernetes and a bit of Rust."
        ),
        personal=PersonalInfo(name="Asha", location=Location(city="Bengaluru")),
        professional=ProfessionalInfo(
            current_title="Data Engineer", current_company="Acme", total_experience_years=6,
        ),
        skills=[
            Skill(name="Python", proficiency=ProficiencyLevel.ADVANCED, years_used=5),
            Skill(name="PostgreSQL"),
        ],
        experience=[
            WorkExperience(title="Data Engineer", company="Acme", start_date="2023-01-01",
                           end_date=None, is_current=True,
                           description="Built batch pipelines on Apache Spark and Airflow."),
            WorkExperience(title="Analyst", company="Beta", start_date="2020-01-01",
                           end_date="2022-12-01", description="Dashboards in SQL."),
        ],
        signals=Signals(skill_assessment_scores={"Python": 81.0}, notice_period_days=90,
                        recruiter_response_rate=0.9, open_to_work=True),
    )
    base.update(overrides)
    return Profile(**base)  # type: ignore[arg-type]


def test_evidence_strength_ladder() -> None:
    p = _profile()
    py = find_evidence("python", p)
    assert [s.strength for s in py][:2] == ["verified", "structured"]
    spark = find_evidence("spark", p)
    assert spark[0].strength == "demonstrated"
    assert spark[0].label.startswith("Data Engineer @ Acme (2023–present)")
    rust = find_evidence("rust", p)
    assert [s.strength for s in rust] == ["mentioned"]
    assert find_evidence("golang", p) == []


def test_span_offsets_point_at_the_term() -> None:
    for span in find_evidence("airflow", _profile()) + find_evidence("kubernetes", _profile()):
        assert span.text[span.start:span.end].lower() in {"airflow", "kubernetes"}


def test_aliases_are_used_but_short_aliases_do_not_match_prose() -> None:
    p = _profile()
    # "sql" aliases include postgresql -> the structured PostgreSQL skill counts
    assert any(s.source == "skills" for s in find_evidence("sql", p))
    # "k8s" folds to kubernetes, found in the summary
    assert find_evidence("k8s", p)[0].strength == "mentioned"
    # "ml" must not match random words; "ai" must not match "Airflow"
    assert find_evidence("ai", p) == []


def test_explanation_points_sum_to_overall_and_title_factor() -> None:
    p = _profile()
    parsed = parse_query("data engineer with spark and rust in bangalore")
    scores: dict[str, float | None] = {
        "skill_match": 0.5, "semantic_similarity": 0.6, "cross_encoder_score": 0.4,
        "behavioral_score": 0.8, "keyword_match": 0.7, "experience_match": 0.6,
        "career_trajectory_score": 0.5, "skill_proficiency_score": 0.3,
        "location_match": None,
    }
    mean = sum(WEIGHTS[d] * v for d, v in scores.items() if v is not None) / sum(WEIGHTS.values())
    overall = mean * 0.8  # executor applied a 0.8 title factor
    ex = explain_candidate(p, parsed, scores, overall, WEIGHTS)
    assert abs(sum(c.points for c in ex.contributions) - overall) < 1e-3
    assert abs(ex.title_factor - 0.8) < 1e-3
    assert ex.contributions == sorted(ex.contributions, key=lambda c: -c.points)

    by_req = {r.requirement: r for r in ex.requirements}
    assert by_req["spark"].status == "met" and by_req["spark"].strength == "demonstrated"
    assert by_req["rust"].status == "partial"
    assert by_req["Bangalore"].status == "met"  # Bengaluru folds to Bangalore
    assert ex.requirements[0].kind == "role" and ex.requirements[0].status == "met"
    assert any("Claimed but unproven: rust" in r for r in ex.risks)
    assert any("90-day notice" in r for r in ex.risks)
    assert any("Replies to 90%" in s for s in ex.strengths)
    assert ex.verdict in {"strong", "good", "potential", "weak"}
    assert "proves 1/2 asked skills" in ex.headline


def test_missing_required_skill_and_experience_band() -> None:
    parsed = parse_query("backend engineer with golang 8-12 years")
    ex = explain_candidate(_profile(), parsed, {"skill_match": 0.0}, 0.2,
                           {"skill_match": 1.0})
    by_kind = {r.kind: r for r in ex.requirements}
    assert by_kind["skill"].status == "missing"
    assert by_kind["experience"].status == "missing"  # 6 yrs vs 8-12
    assert any("No evidence of required skill" in r for r in ex.risks)


def test_trajectory_modes_differ_only_by_the_open_role() -> None:
    p = _profile()
    a = trajectory_facts(p, include_current_tenure=False, as_of="2026-01-01")
    b = trajectory_facts(p, include_current_tenure=True, as_of="2026-01-01")
    assert a["avg_tenure_months"] == 35  # only the 2020-2022 role
    assert a["current_tenure_months"] == b["current_tenure_months"] == 36
    assert b["avg_tenure_months"] == 35.5
    assert (a["mode"], b["mode"]) == ("completed_roles", "include_current")


def test_short_skills_need_a_listed_or_assessed_match_not_prose() -> None:
    p = _profile(
        raw_text="Summary: Drove go-live of CRM two years ago; built a C-level deck, R&D liaison.",
        skills=[Skill(name="JS"), Skill(name="ML")],
        experience=[WorkExperience(title="PM", company="A",
                                   description="Drove go-live; R&D and C-level reviews.")],
    )
    for term in ("go", "r", "c"):
        assert find_evidence(term, p) == [], term
    # short aliases still match the skills section exactly
    assert find_evidence("javascript", p)[0].source == "skills"
    assert find_evidence("machine learning", p)[0].source == "skills"


def test_trajectory_facts_only_treat_flagged_roles_as_current() -> None:
    p = _profile(experience=[
        WorkExperience(title="A", company="X", start_date="2020-01", end_date=None,
                       is_current=False),
    ])
    assert trajectory_facts(p, True, as_of="2026-01")["current_tenure_months"] is None
