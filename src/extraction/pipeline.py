from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.extraction.company import extract_company
from src.extraction.domain import extract_industry
from src.extraction.experience_years import extract_experience_years
from src.extraction.seniority import extract_seniority
from src.extraction.title import extract_title


@dataclass
class ExtractionResult:
    value: Any
    source: str
    confidence: str = "medium"


@dataclass
class ExtractionBundle:
    current_title: ExtractionResult = field(default_factory=lambda: ExtractionResult(None, "not_found"))
    current_company: ExtractionResult = field(default_factory=lambda: ExtractionResult(None, "not_found"))
    total_experience_years: ExtractionResult = field(default_factory=lambda: ExtractionResult(None, "not_found"))
    industry: ExtractionResult = field(default_factory=lambda: ExtractionResult(None, "not_found"))
    seniority_level: ExtractionResult = field(default_factory=lambda: ExtractionResult(None, "not_found"))


class FieldExtractorPipeline:
    def extract(self, raw: dict[str, Any]) -> ExtractionBundle:
        prof = raw.get("profile", {})
        history = raw.get("career_history", [])
        skills = raw.get("skills", [])

        title_val, title_src = extract_title(prof, history)
        company_val, company_src = extract_company(prof, history)
        years_val, years_src = extract_experience_years(prof, history)
        industry_val, industry_src = extract_industry(prof, skills, history)
        seniority_val, seniority_src = extract_seniority(title_val, years_val, history)

        return ExtractionBundle(
            current_title=ExtractionResult(title_val, title_src),
            current_company=ExtractionResult(company_val, company_src),
            total_experience_years=ExtractionResult(years_val, years_src),
            industry=ExtractionResult(industry_val, industry_src),
            seniority_level=ExtractionResult(seniority_val, seniority_src),
        )
