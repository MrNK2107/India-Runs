from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ProfileSource(StrEnum):
    LINKEDIN = "linkedin"
    NAUKRI = "naukri"
    GITHUB = "github"
    RESUME_PDF = "resume_pdf"
    CAREER_PAGE = "career_page"
    MANUAL = "manual"
    REDROB = "redrob"


class SkillCategory(StrEnum):
    PROGRAMMING_LANGUAGE = "programming_language"
    FRAMEWORK = "framework"
    TOOL = "tool"
    SOFT_SKILL = "soft_skill"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    CERTIFICATION = "certification"


class ProficiencyLevel(StrEnum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SkillImportance(StrEnum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    NICE_TO_HAVE = "nice_to_have"


class EmploymentType(StrEnum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    STUDENT = "student"


class MatchRecommendation(StrEnum):
    STRONG = "strong_match"
    GOOD = "good_match"
    POTENTIAL = "potential_match"
    WEAK = "weak_match"


class SearchMethod(StrEnum):
    HYBRID = "hybrid"
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"


class Location(BaseModel):
    city: str | None = None
    state: str | None = None
    country: str = "India"
    is_remote_ok: bool = False


class PersonalInfo(BaseModel):
    name: str
    location: Location = Field(default_factory=Location)
    languages_spoken: list[str] = Field(default_factory=list)
    native_language: str | None = None


class ProfessionalInfo(BaseModel):
    current_title: str | None = None
    current_company: str | None = None
    total_experience_years: float | None = None
    industry: str | None = None
    employment_type: EmploymentType | None = None
    seniority_level: int | None = None


class Skill(BaseModel):
    name: str
    category: SkillCategory = SkillCategory.TOOL
    proficiency: ProficiencyLevel | None = None
    years_used: float | None = None
    evidence: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class WorkExperience(BaseModel):
    title: str
    company: str
    start_date: str | None = None
    end_date: str | None = None
    is_current: bool = False
    description: str = ""
    highlights: list[str] = Field(default_factory=list)
    skills_demonstrated: list[str] = Field(default_factory=list)
    location: str | None = None


class Education(BaseModel):
    institution: str
    degree: str | None = None
    field: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    gpa: float | None = None


class Signals(BaseModel):
    """Behavioral signals from the Redrob platform — 20+ dimensions."""
    is_passive: bool = False
    last_active_date: str | None = None
    open_to_work: bool | None = None
    github_activity_score: float | None = None
    has_portfolio: bool = False
    certifications: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)
    speaking_engagements: list[str] = Field(default_factory=list)
    # Full redrob_signals enrichment
    profile_completeness_score: float | None = None
    recruiter_response_rate: float | None = None
    avg_response_time_hours: float | None = None
    saved_by_recruiters_30d: int | None = None
    profile_views_received_30d: int | None = None
    applications_submitted_30d: int | None = None
    connection_count: int | None = None
    endorsements_received: int | None = None
    search_appearance_30d: int | None = None
    interview_completion_rate: float | None = None
    offer_acceptance_rate: float | None = None
    notice_period_days: int | None = None
    preferred_work_mode: str | None = None
    willing_to_relocate: bool | None = None
    verified_email: bool | None = None
    verified_phone: bool | None = None
    expected_salary_min: float | None = None
    expected_salary_max: float | None = None
    linkedin_connected: bool | None = None
    skill_assessment_scores: dict[str, float] = Field(default_factory=dict)


class ProfileMetadata(BaseModel):
    language_detected: str = "en"
    original_language: str = "en"
    was_translated: bool = False
    translation_confidence: float | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    data_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)


class Profile(BaseModel):
    profile_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: ProfileSource = ProfileSource.MANUAL
    raw_text: str = ""
    personal: PersonalInfo
    professional: ProfessionalInfo = Field(default_factory=ProfessionalInfo)
    skills: list[Skill] = Field(default_factory=list)
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    signals: Signals = Field(default_factory=Signals)
    metadata: ProfileMetadata = Field(default_factory=ProfileMetadata)


class RequiredSkill(BaseModel):
    name: str
    importance: SkillImportance = SkillImportance.REQUIRED
    min_proficiency: ProficiencyLevel | None = None
    min_years: float | None = None


class PreferredSkill(BaseModel):
    name: str
    importance: SkillImportance = SkillImportance.NICE_TO_HAVE
    weight: float = Field(default=0.5, ge=0.0, le=1.0)


class ExperienceRequirements(BaseModel):
    min_years: float | None = None
    max_years: float | None = None
    industry: str | None = None


class LocationRequirements(BaseModel):
    city: str | None = None
    state: str | None = None
    country: str | None = None
    remote_ok: bool = False
    hybrid_ok: bool = False


class EducationRequirements(BaseModel):
    min_degree: str | None = None
    field: str | None = None


class SalaryRequirements(BaseModel):
    min: float | None = None
    max: float | None = None
    currency: str = "INR"


class QueryFilters(BaseModel):
    exclude_companies: list[str] = Field(default_factory=list)
    include_companies: list[str] = Field(default_factory=list)
    must_have_certifications: list[str] = Field(default_factory=list)
    languages_required: list[str] = Field(default_factory=list)


class ParsedQuery(BaseModel):
    required_skills: list[RequiredSkill] = Field(default_factory=list)
    preferred_skills: list[PreferredSkill] = Field(default_factory=list)
    subskills: dict[str, list[str]] = Field(default_factory=dict)
    experience: ExperienceRequirements = Field(default_factory=ExperienceRequirements)
    location: LocationRequirements = Field(default_factory=LocationRequirements)
    education: EducationRequirements = Field(default_factory=EducationRequirements)
    salary: SalaryRequirements = Field(default_factory=SalaryRequirements)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    original_query: str = ""


class JobQuery(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_query: str
    parsed: ParsedQuery = Field(default_factory=ParsedQuery)
    language: str = "en"


class MatchScores(BaseModel):
    overall: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    keyword_match: float = Field(default=0.0, ge=0.0, le=1.0)
    skill_match: float = Field(default=0.0, ge=0.0, le=1.0)
    experience_match: float = Field(default=0.0, ge=0.0, le=1.0)
    location_match: float | None = Field(default=None, ge=0.0, le=1.0)
    education_match: float | None = Field(default=None, ge=0.0, le=1.0)
    cross_encoder_score: float | None = Field(default=None, ge=0.0, le=1.0)
    behavioral_score: float | None = Field(default=None, ge=0.0, le=1.0)
    career_trajectory_score: float | None = Field(default=None, ge=0.0, le=1.0)
    skill_proficiency_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SkillDetail(BaseModel):
    skill: str
    required: bool
    found: bool
    proficiency_match: bool
    evidence: str = ""


class Rationale(BaseModel):
    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    skill_details: list[SkillDetail] = Field(default_factory=list)
    experience_analysis: str = ""
    recommendation: MatchRecommendation = MatchRecommendation.GOOD


class MatchMetadata(BaseModel):
    search_method: SearchMethod = SearchMethod.HYBRID
    reranked: bool = False
    language_matched: bool = False
    passive_candidate: bool = False
    processing_time_ms: int = 0
    translation_fallback: bool = False


class MatchResult(BaseModel):
    match_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str
    profile_id: str
    rank: int
    name: str = ""
    current_title: str | None = None
    current_company: str | None = None
    location: str | None = None
    experience_years: float | None = None
    scores: MatchScores
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    rationale: Rationale = Field(default_factory=Rationale)
    metadata: MatchMetadata = Field(default_factory=MatchMetadata)


class SearchFilters(BaseModel):
    location: str | None = None
    min_experience_years: float | None = None
    max_experience_years: float | None = None
    remote_ok: bool = False
    exclude_companies: list[str] = Field(default_factory=list)
    include_companies: list[str] = Field(default_factory=list)


# UI/API slider dimensions (percent weights 0-100); must match matching.scorer.DIM_TO_ACTUAL.
SLIDER_DIMENSIONS: tuple[str, ...] = (
    "skill_match", "experience_match", "education_match",
    "assessment_score", "behavioral_signals", "cultural_fit",
)


class SearchRequest(BaseModel):
    """API request. Stricter than SearchFilters itself, which the executor also builds
    from LLM-parsed queries and must stay permissive."""

    query: str = Field(min_length=1, max_length=2000)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    max_results: int = Field(default=10, ge=1, le=100)
    include_rationale: bool = True
    language: str | None = None
    use_turbo: bool = False
    slider_weights: dict[str, float] = Field(default_factory=dict)
    # "completed_roles" (default, the submitted ranking): job-hopping/stability uses
    # finished roles only. "include_current": the open role's tenure counts too.
    trajectory_mode: Literal["completed_roles", "include_current"] = "completed_roles"
    # "listed" (default, the submitted ranking): the skills score counts the skills
    # section only. "work_history": skills proven in a role or assessment count too.
    skill_evidence: Literal["listed", "work_history"] = "listed"
    # Attach a per-candidate Explanation (score contributions + evidence spans).
    explain: bool = True

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped

    @field_validator("slider_weights")
    @classmethod
    def _check_slider_weights(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(v) - set(SLIDER_DIMENSIONS))
        if unknown:
            raise ValueError(
                f"unknown slider_weights key(s) {unknown}; valid keys: {list(SLIDER_DIMENSIONS)}"
            )
        for key, weight in v.items():
            if not math.isfinite(weight) or not 0 <= weight <= 100:
                raise ValueError(
                    f"slider_weights[{key!r}] must be a number between 0 and 100, got {weight}"
                )
        return v

    @model_validator(mode="after")
    def _check_experience_filters(self) -> SearchRequest:
        lo = self.filters.min_experience_years
        hi = self.filters.max_experience_years
        for name, val in (("min_experience_years", lo), ("max_experience_years", hi)):
            if val is not None and (not math.isfinite(val) or val < 0):
                raise ValueError(f"filters.{name} must be a non-negative number, got {val}")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                f"filters.min_experience_years ({lo}) must not exceed "
                f"filters.max_experience_years ({hi})"
            )
        return self


class EvidenceSpan(BaseModel):
    """A quoted piece of the profile that supports (or is the only support for) a requirement."""

    term: str
    source: Literal["assessment", "skills", "experience", "summary", "headline", "education"]
    label: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    strength: Literal["verified", "structured", "demonstrated", "mentioned"]


class RequirementCheck(BaseModel):
    requirement: str
    kind: Literal["skill", "location", "experience", "role"]
    status: Literal["met", "partial", "missing"]
    strength: Literal["verified", "structured", "demonstrated", "mentioned"] | None = None
    detail: str = ""
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class ScoreContribution(BaseModel):
    dimension: str
    label: str
    score: float
    weight: float
    points: float  # share of the overall score this dimension contributed (0-1 scale)


class Explanation(BaseModel):
    profile_id: str
    verdict: Literal["strong", "good", "potential", "weak"]
    headline: str
    overall: float
    title_factor: float = 1.0
    contributions: list[ScoreContribution] = Field(default_factory=list)
    requirements: list[RequirementCheck] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    trajectory: dict[str, float | int | str | None] = Field(default_factory=dict)


class ExplainRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    profile_id: str = Field(min_length=1, max_length=200)
    slider_weights: dict[str, float] = Field(default_factory=dict)
    trajectory_mode: Literal["completed_roles", "include_current"] = "completed_roles"
    skill_evidence: Literal["listed", "work_history"] = "listed"

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class SearchResultItem(BaseModel):
    rank: int
    profile_id: str
    name: str
    current_title: str | None = None
    current_company: str | None = None
    location: str | None = None
    experience_years: float | None = None
    scores: MatchScores
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    rationale: Rationale
    passive_candidate: bool = False
    language_matched: bool = False
    explanation: Explanation | None = None


class SearchMetadata(BaseModel):
    methods_used: list[str] = Field(default_factory=list)
    replan_count: int = 0
    total_time_ms: int = 0
    listwise_ranked: bool = False
    pii_anonymized: bool = True
    total_profiles_in_index: int = 0


class SearchResponse(BaseModel):
    query_id: str
    total_candidates_searched: int
    results: list[SearchResultItem] = Field(default_factory=list)
    message: str | None = None
    suggestions: list[str] = Field(default_factory=list)
    processing_time_ms: int = 0
    search_metadata: SearchMetadata = Field(default_factory=SearchMetadata)


class IngestResponse(BaseModel):
    total_profiles: int
    successful: int
    failed: int
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    profile_ids: list[str] = Field(default_factory=list)
    language_distribution: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    index_size: int
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    engine: dict[str, str] = Field(default_factory=dict)
    last_updated: str | None = None
