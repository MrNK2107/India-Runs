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
    "Output valid JSON matching this schema:\n"
    '{\n'
    '  "required_skills": [{"name": "...", "importance": "required",\n'
    '    "min_proficiency": "...", "min_years": null}],\n'
    '  "preferred_skills": [{"name": "...", "importance": "nice_to_have", "weight": 0.5}],\n'
    '  "experience": {"min_years": null, "max_years": null, "industry": null},\n'
    '  "location": {"city": null, "state": null, "country": null,\n'
    '    "remote_ok": false, "hybrid_ok": false},\n'
    '  "education": {"min_degree": null, "field": null},\n'
    '  "filters": {"exclude_companies": [], "include_companies": [],\n'
    '    "must_have_certifications": [], "languages_required": []}\n'
    '}\n\n'
    "If the query is ambiguous, make reasonable assumptions and note them.\n"
    "Output ONLY valid JSON, no other text."
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

RATIONALE_SYSTEM_PROMPT = (
    "You are generating a candidate evaluation report for a recruiter.\n\n"
    "JOB REQUIREMENTS:\n"
    "{job_requirements_json}\n\n"
    "CANDIDATE PROFILE:\n"
    "{candidate_profile_summary}\n\n"
    "MATCH SCORES:\n"
    "{scores_json}\n\n"
    "Generate a detailed rationale report. Be specific \u2014 reference actual companies, "
    "roles, and skills from the profile. Do not make generic statements.\n\n"
    "Requirements:\n"
    "- summary: 2-3 sentences, specific to this candidate\n"
    "- strengths: list specific matches with evidence\n"
    "- gaps: list specific concerns or missing requirements\n"
    "- skill_details: for each required skill, note if found and the evidence\n"
    "- experience_analysis: paragraph about work history relevance\n"
    "- recommendation: one of strong_match, good_match, potential_match, weak_match\n\n"
    "Output valid JSON only."
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
