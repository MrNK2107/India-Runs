RATIONALE_TEMPLATE = """Generate a candidate evaluation report for a recruiter.

JOB REQUIREMENTS:
{job_requirements}

CANDIDATE PROFILE:
Name: {name}
Title: {current_title} at {current_company}
Experience: {experience_years} years
Skills: {skill_names}
Location: {location}

MATCH SCORES:
- Overall: {overall_score:.2f}
- Skill Match: {skill_score:.2f}
- Experience Match: {experience_score:.2f}
- Semantic Match: {semantic_score:.2f}

Generate a JSON response with:
- summary: 2-3 sentence overview
- strengths: list of specific strengths with evidence
- gaps: list of specific gaps
- skill_details: for each required skill, note found/evidence
- experience_analysis: paragraph about work history relevance
- recommendation: strong_match | good_match | potential_match | weak_match"""

SKILL_EVIDENCE_TEMPLATE = """Candidate skill: {skill_name} (proficiency: {proficiency})
Evidence: {evidence}
Required level: {required_level}
Match: {matched}"""
