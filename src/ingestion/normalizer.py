from __future__ import annotations

import re
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

    professional = ProfessionalInfo(
        current_title=prof.get("current_title"),
        current_company=prof.get("current_company"),
        total_experience_years=prof.get("years_of_experience"),
        industry=prof.get("current_industry"),
        employment_type=None,
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
            location=entry.get("industry"),
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

    raw_text_parts = [
        prof.get("headline", ""),
        prof.get("summary", ""),
    ]
    for exp in experience:
        raw_text_parts.append(exp.description)
    raw_text = "\n\n".join(p for p in raw_text_parts if p)

    if native_langs:
        personal.native_language = native_langs[0]
    personal.languages_spoken = languages

    return Profile(
        profile_id=raw.get("candidate_id", ""),
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


def _build_signals(rs: dict[str, Any], certs: list[str]) -> Signals:
    return Signals(
        is_passive=not rs.get("open_to_work_flag", True),
        last_active_date=rs.get("last_active_date"),
        open_to_work=rs.get("open_to_work_flag"),
        github_activity_score=rs.get("github_activity_score"),
        certifications=certs,
        has_portfolio=rs.get("linkedin_connected", False),
    )
