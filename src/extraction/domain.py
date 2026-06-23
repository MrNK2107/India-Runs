from __future__ import annotations

from typing import Any

_INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "fintech": ["fintech", "banking", "finance", "payment", "insurance", "investment"],
    "healthcare": ["healthcare", "health", "medical", "clinical", "pharma", "biotech"],
    "ecommerce": ["ecommerce", "e-commerce", "retail", "marketplace", "consumer internet"],
    "ai/ml": ["machine learning", "artificial intelligence", "deep learning", "nlp",
              "llm", "computer vision", "mlops"],
    "edtech": ["edtech", "education", "elearning", "learning", "online education"],
    "saas": ["saas", "b2b", "enterprise software", "cloud software"],
    "infrastructure": ["devops", "infrastructure", "cloud", "kubernetes", "docker",
                        "terraform", "platform engineering"],
    "data": ["data engineering", "data science", "data analytics", "big data", "data pipeline"],
    "cybersecurity": ["cybersecurity", "security", "infosec", "penetration testing"],
}

_COMPANY_INDUSTRY: dict[str, str] = {
    "mindtree": "it_services",
    "infosys": "it_services",
    "tcs": "it_services",
    "wipro": "it_services",
    "accenture": "it_services",
    "google": "internet",
    "amazon": "ecommerce",
    "microsoft": "saas",
    "flipkart": "ecommerce",
    "swiggy": "ecommerce",
    "zomato": "ecommerce",
    "razorpay": "fintech",
    "phonepe": "fintech",
    "paytm": "fintech",
    "byjus": "edtech",
    "unacademy": "edtech",
}


def extract_industry(
    prof: dict[str, Any],
    skills: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> tuple[str | None, str]:
    direct = prof.get("current_industry")
    if direct and isinstance(direct, str) and direct.strip():
        return direct.strip(), "direct"

    for entry in history:
        company = (entry.get("company") or "").lower().strip()
        if company in _COMPANY_INDUSTRY:
            return _COMPANY_INDUSTRY[company], "company_map"
        for known_company, mapped_industry in _COMPANY_INDUSTRY.items():
            if known_company in company:
                return mapped_industry, "company_map"

    skill_names = [s.get("name", "") for s in skills]
    all_text = " ".join(skill_names).lower()
    for industry, keywords in _INDUSTRY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in all_text:
                return industry, "skills"

    headline = prof.get("headline", "")
    summary = prof.get("summary", "")
    combined = f"{headline} {summary}".lower()
    for industry, keywords in _INDUSTRY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined:
                return industry, "headline_summary"

    return None, "not_found"
