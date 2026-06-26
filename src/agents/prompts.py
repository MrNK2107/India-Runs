import logging

logger = logging.getLogger(__name__)


def fill_placeholders(template: str, **kwargs: str) -> str:
    """Substitute placeholders in a template, warning about unfilled ones."""
    result = template
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, value)
    unfilled = _find_unfilled_placeholders(result)
    for u in unfilled:
        logger.warning("Unfilled placeholder in prompt template: {%s}", u)
        result = result.replace("{" + u + "}", "")
    return result


def _find_unfilled_placeholders(text: str) -> list[str]:
    import re
    return re.findall(r"\{(\w+)\}", text)


PLANNER_SYSTEM_PROMPT = (
    "You are an expert recruiter's assistant. Given a natural language job query, "
    "parse it into a structured search specification.\n\n"
    "Extract:\n"
    "- Required skills (with importance: required/preferred/nice_to_have)\n"
    "- Experience requirements (years, industry)\n"
    "- Location preferences (city, remote preference)\n"
    "- Education requirements\n"
    "- Any exclusion criteria\n\n"
    "CRITICAL INSTRUCTIONS FOR SKILL EXTRACTION:\n"
    "1. Do NOT literally extract vague high-level concepts, job titles, or abstract goals as skills. "  # noqa: E501
    "For example, do NOT extract 'Full-Stack Development', 'Full-Stack Developer', 'Web Transformation', "  # noqa: E501
    "'Modern Standards', 'Cloud Engineer', 'Software Engineer', or 'Database Developer' as skills.\n"  # noqa: E501
    "2. Instead, map these abstract job-role terms into the concrete technical skills, languages, "  # noqa: E501
    "frameworks, or databases that real candidates would list in their profile skills array. For example:\n"  # noqa: E501
    "   - 'Full-Stack Developer' -> map to ['React', 'Node.js', 'JavaScript', 'HTML', 'CSS']\n"  # noqa: E501
    "   - 'Cloud Engineer' / 'DevOps' -> map to ['AWS', 'Terraform', 'Docker', 'Kubernetes', 'CI/CD']\n"  # noqa: E501
    "   - 'Backend Developer' -> map to ['Python', 'Django', 'FastAPI', 'Node.js', 'SQL', 'PostgreSQL']\n"  # noqa: E501
    "   - 'Frontend Developer' -> map to ['React', 'TypeScript', 'Next.js', 'HTML', 'CSS', 'JavaScript']\n"  # noqa: E501
    "3. Use standard, commonly-used skill names that candidates actually list.\n\n"
    "Output ONLY valid JSON, no other text. "
    "Do NOT wrap in markdown code fences. Return raw JSON only.\n"
    "Schema:\n"
    '{\n'
    '  "required_skills": [{"name": "...", "importance": "required",\n'
    '    "min_proficiency": null, "min_years": null}],\n'
    '  "preferred_skills": [],\n'
    '  "experience": {"min_years": null, "max_years": null, "industry": null},\n'
    '  "location": {"city": null, "state": null, "country": null,\n'
    '    "remote_ok": false, "hybrid_ok": false},\n'
    '  "education": {"min_degree": null, "field": null},\n'
    '  "filters": {"exclude_companies": [], "include_companies": [],\n'
    '    "must_have_certifications": [], "languages_required": []}\n'
    '}\n'
)

REFLECTOR_SYSTEM_PROMPT = (
    "You are a critical hiring evaluator. For each candidate in the search results, "
    "assess whether they truly match the job requirements.\n\n"
    "For each candidate, provide:\n"
    '1. overall_assessment: "strong_match" | "good_match" | "potential_match" | "weak_match"\n'
    "2. key_strengths: list of specific reasons why they match\n"
    "3. key_gaps: list of specific reasons why they might not match\n"
    "4. concerns: any red flags or uncertainties\n"
    "5. should_keep: boolean\n\n"
    'Be strict — a "strong_match" means you would confidently shortlist this person.\n'
    'A "good_match" means they could work with some caveats.\n'
    'A "potential_match" means they are worth a phone screen.\n'
    'A "weak_match" means they should be dropped.\n\n'
    "Output valid JSON array. No other text."
)

REPLAN_SYSTEM_PROMPT = (
    "The previous search did not yield enough strong matches.\n"
    "Original query: {original_query}\n"
    "Previous parsed parameters: {previous_params}\n"
    "Reflector feedback: {feedback}\n\n"
    "Revise the search parameters based on the feedback.\n"
    "Common revisions:\n"
    "- Broaden skill requirements (move some required to preferred)\n"
    "- Relax experience requirements\n"
    "- Expand location (remove city filter, allow remote)\n"
    "- Remove company exclusions\n\n"
    "Output the revised parsed query as valid JSON (same schema as before)."
)
