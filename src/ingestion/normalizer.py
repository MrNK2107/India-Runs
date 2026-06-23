from __future__ import annotations

import re
import uuid
from typing import Any

from src.core.models import (
    Education,
    Location,
    PersonalInfo,
    ProfessionalInfo,
    ProficiencyLevel,
    Profile,
    ProfileMetadata,
    ProfileSource,
    Signals,
    Skill,
    WorkExperience,
)
from src.extraction.pipeline import FieldExtractorPipeline

_PROFICIENCY_MAP = {
    "beginner": ProficiencyLevel.BEGINNER,
    "intermediate": ProficiencyLevel.INTERMEDIATE,
    "advanced": ProficiencyLevel.ADVANCED,
    "expert": ProficiencyLevel.EXPERT,
}

def _parse_grade(grade: str | None) -> float | None:
    if not grade:
        return None
    grade = grade.strip()
    m = re.search(r"([\d.]+)", grade)
    if m:
        val = float(m.group(1))
        if "%" in grade:
            val = val / 100.0 * 10.0
        return round(val, 2)
    return None


def normalize_redrob(raw: dict[str, Any], source: str = "redrob") -> Profile:
    prof = raw.get("profile", {})

    city_state = prof.get("location", "") or ""
    city_parts = city_state.split(",")
    city = city_parts[0].strip() if city_parts else None
    state = city_parts[1].strip() if len(city_parts) > 1 else None

    location = Location(
        city=city or None,
        state=state or None,
        country=prof.get("country", "India"),
        is_remote_ok=False,
    )

    personal = PersonalInfo(
        name=prof.get("anonymized_name", ""),
        location=location,
        languages_spoken=[],
        native_language=None,
    )

    extracted = FieldExtractorPipeline().extract(raw)

    professional = ProfessionalInfo(
        current_title=extracted.current_title.value or prof.get("current_title"),
        current_company=extracted.current_company.value or prof.get("current_company"),
        total_experience_years=(
            extracted.total_experience_years.value or prof.get("years_of_experience")
        ),
        industry=extracted.industry.value or prof.get("current_industry"),
        employment_type=None,
        seniority_level=extracted.seniority_level.value,
    )

    skills = [
        Skill(
            name=s["name"],
            category=_infer_skill_category(s["name"]),
            proficiency=_PROFICIENCY_MAP.get(s.get("proficiency", "")),
            years_used=(
                round(s.get("duration_months", 0) / 12, 1)
                if s.get("duration_months")
                else None
            ),
            evidence=None,
            confidence=min(1.0, s.get("endorsements", 0) / 50),
        )
        for s in raw.get("skills", [])
    ]

    experience = [
        WorkExperience(
            title=entry["title"],
            company=entry["company"],
            start_date=entry.get("start_date"),
            end_date=entry.get("end_date"),
            is_current=entry.get("is_current", False),
            description=entry.get("description", ""),
            location=entry.get("location"),
        )
        for entry in raw.get("career_history", [])
    ]

    education = [
        Education(
            institution=edu["institution"],
            degree=edu.get("degree"),
            field=edu.get("field_of_study"),
            start_date=str(edu.get("start_year", "")) if edu.get("start_year") else None,
            end_date=str(edu.get("end_year", "")) if edu.get("end_year") else None,
            gpa=_parse_grade(edu.get("grade")),
        )
        for edu in raw.get("education", [])
    ]

    certs = [c["name"] for c in raw.get("certifications", [])]

    languages = [
        lang["language"] for lang in raw.get("languages", [])
    ]
    native_langs = [
        lang["language"]
        for lang in raw.get("languages", [])
        if lang.get("proficiency") == "native"
    ]

    signals = _build_signals(raw.get("redrob_signals", {}), certs)

    raw_text = _build_raw_text(prof, experience, skills, education, certs, languages)

    from src.matching.skill_matcher import SKILL_ALIASES as _KNOWN_ALIASES
    existing_skill_names = {s.name.lower() for s in skills}
    raw_lower = raw_text.lower()
    for _canon, _aliases in _KNOWN_ALIASES.items():
        if _canon not in existing_skill_names and _canon in raw_lower:
            skills.append(
                Skill(
                    name=_canon.title(),
                    category=_infer_skill_category(_canon),
                    evidence="Extracted from profile text",
                    confidence=0.6,
                )
            )
            existing_skill_names.add(_canon)

    if native_langs:
        personal.native_language = native_langs[0]
    personal.languages_spoken = languages

    return Profile(
        profile_id=raw.get("candidate_id") or str(uuid.uuid4()),
        source=ProfileSource.REDROB,
        raw_text=raw_text,
        personal=personal,
        professional=professional,
        skills=skills,
        experience=experience,
        education=education,
        signals=signals,
        metadata=ProfileMetadata(
            language_detected="en",
            data_quality_score=0.0,
        ),
    )


def _infer_skill_category(name: str) -> Any:
    from src.core.models import SkillCategory

    name_lower = name.lower()
    lang_keywords = [
        "python", "java", "javascript", "typescript", "go", "rust",
        "c++", "c#", "ruby", "php", "swift", "kotlin", "scala",
        "r", "sql", "html", "css", "bash",
    ]
    framework_keywords = [
        "react", "angular", "vue", "django", "flask", "spring",
        "pytorch", "tensorflow", "keras", "spark", "hadoop",
        "node.js", "express", "next.js", "tailwind", "redux",
    ]
    tool_keywords = [
        "docker", "kubernetes", "aws", "gcp", "azure", "terraform",
        "airflow", "kafka", "git", "jenkins", "ansible",
        "snowflake", "bigquery", "mongodb", "redis", "postgresql",
    ]

    if any(kw in name_lower for kw in lang_keywords):
        return SkillCategory.PROGRAMMING_LANGUAGE
    if any(kw in name_lower for kw in framework_keywords):
        return SkillCategory.FRAMEWORK
    if any(kw in name_lower for kw in tool_keywords):
        return SkillCategory.TOOL

    domain_keywords = [
        "machine learning", "deep learning", "nlp", "computer vision",
        "data science", "data engineering", "mlops",
        "natural language processing", "llm", "rag",
        "statistical", "analytics",
    ]
    if any(kw in name_lower for kw in domain_keywords):
        return SkillCategory.DOMAIN_KNOWLEDGE

    return SkillCategory.TOOL


def _build_raw_text(
    prof: dict[str, Any],
    experience: list[WorkExperience],
    skills: list[Skill],
    education: list[Education],
    certs: list[str],
    languages: list[str],
) -> str:
    name = prof.get("anonymized_name", "")
    title = prof.get("current_title", "")
    company = prof.get("current_company", "")
    summary = prof.get("summary", "")
    skill_names = ", ".join(s.name for s in skills)
    exp_strs = [
        f"{e.title} at {e.company} ({e.start_date or ''}-{e.end_date or ''}): {e.description}"
        for e in experience
    ]
    exp_text = "; ".join(exp_strs) if exp_strs else ""
    edu_strs = [
        f"{e.degree or ''} in {e.field or ''} at {e.institution}"
        for e in education
    ]
    edu_text = "; ".join(edu_strs) if edu_strs else ""
    certs_text = ", ".join(certs) if certs else ""
    langs_text = ", ".join(languages) if languages else ""

    parts = [
        f"Name: {name}.",
        f"Title: {title}." if title else "",
        f"Company: {company}." if company else "",
        f"Summary: {summary}." if summary else "",
        f"Skills: {skill_names}." if skill_names else "",
        f"Experience: {exp_text}." if exp_text else "",
        f"Education: {edu_text}." if edu_text else "",
        f"Certifications: {certs_text}." if certs_text else "",
        f"Languages: {langs_text}." if langs_text else "",
    ]
    return " ".join(p for p in parts if p)


def _build_signals(rs: dict[str, Any], certs: list[str]) -> Signals:
    salary_range = rs.get("expected_salary_range_inr_lpa", {}) or {}
    return Signals(
        is_passive=not rs.get("open_to_work_flag", True),
        last_active_date=rs.get("last_active_date"),
        open_to_work=rs.get("open_to_work_flag"),
        github_activity_score=rs.get("github_activity_score"),
        certifications=certs,
        has_portfolio=rs.get("linkedin_connected", False),
        # Full platform signals
        profile_completeness_score=rs.get("profile_completeness_score"),
        recruiter_response_rate=rs.get("recruiter_response_rate"),
        avg_response_time_hours=rs.get("avg_response_time_hours"),
        saved_by_recruiters_30d=rs.get("saved_by_recruiters_30d"),
        profile_views_received_30d=rs.get("profile_views_received_30d"),
        applications_submitted_30d=rs.get("applications_submitted_30d"),
        connection_count=rs.get("connection_count"),
        endorsements_received=rs.get("endorsements_received"),
        search_appearance_30d=rs.get("search_appearance_30d"),
        interview_completion_rate=rs.get("interview_completion_rate"),
        offer_acceptance_rate=rs.get("offer_acceptance_rate"),
        notice_period_days=rs.get("notice_period_days"),
        preferred_work_mode=rs.get("preferred_work_mode"),
        willing_to_relocate=rs.get("willing_to_relocate"),
        verified_email=rs.get("verified_email"),
        verified_phone=rs.get("verified_phone"),
        expected_salary_min=salary_range.get("min"),
        expected_salary_max=salary_range.get("max"),
        linkedin_connected=rs.get("linkedin_connected"),
        skill_assessment_scores=rs.get("skill_assessment_scores", {}),
    )
