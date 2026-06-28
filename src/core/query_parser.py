from __future__ import annotations

import re
from src.core.constants import INDIAN_CITIES
from src.core.models import (
    ExperienceRequirements,
    LocationRequirements,
    ParsedQuery,
    PreferredSkill,
    QueryFilters,
    RequiredSkill,
    SkillImportance,
)
from src.matching.skill_matcher import SKILL_ALIASES


_SINGLE_WORD_SKILLS: set[str] = set()
_MULTI_WORD_SKILLS: set[str] = set()
_MULTI_WORD_MAP: dict[str, str] = {}

for skill in SKILL_ALIASES:
    norm = skill.strip().lower()
    word_count = len(norm.split())
    if word_count == 1:
        _SINGLE_WORD_SKILLS.add(norm)
    else:
        _MULTI_WORD_SKILLS.add(norm)
        _MULTI_WORD_MAP[norm] = skill
    for alias in SKILL_ALIASES[skill]:
        alias_norm = alias.strip().lower()
        alias_words = len(alias_norm.split())
        if alias_words == 1:
            _SINGLE_WORD_SKILLS.add(alias_norm)
        else:
            _MULTI_WORD_SKILLS.add(alias_norm)
            _MULTI_WORD_MAP[alias_norm] = skill

_EXTRA_SKILLS: set[str] = {
    "python", "java", "javascript", "typescript", "golang", "go", "rust",
    "c++", "c#", "ruby", "php", "scala", "kotlin", "swift", "dart",
    "html", "css", "sass", "less", "webpack", "vite", "babel",
    "node.js", "node", "express", "nestjs", "graphql", "grpc",
    "react", "vue", "angular", "svelte", "next.js", "nuxt", "gatsby",
    "tailwind", "bootstrap", "material ui", "shadcn", "chakra",
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "couchbase", "mariadb", "sqlite",
    "aws", "gcp", "azure", "cloud", "terraform", "pulumi",
    "docker", "kubernetes", "k8s", "helm", "istio", "envoy",
    "ci/cd", "jenkins", "github actions", "gitlab ci", "circleci",
    "argocd", "prometheus", "grafana", "datadog", "new relic",
    "kafka", "rabbitmq", "pulsar", "nats",
    "flask", "django", "fastapi", "spring boot", "spring",
    "pytorch", "tensorflow", "keras", "jax", "scikit-learn",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "spark", "hadoop", "airflow", "prefect", "dbt", "snowflake",
    "bigquery", "redshift", "databricks", "lakehouse",
    "machine learning", "deep learning", "reinforcement learning",
    "natural language processing", "nlp", "computer vision",
    "llm", "large language model", "rag", "langchain",
    "gen ai", "generative ai", "diffusion", "transformer",
    "selenium", "cypress", "playwright", "puppeteer",
    "pytest", "jest", "mocha", "chai", "vitest",
    "rest api", "restful", "rest", "graphql",
    "microservices", "distributed systems", "system design",
    "android", "ios", "flutter", "react native", "kotlin",
    "mlops", "devops", "data engineering", "data science",
    "cybersecurity", "penetration testing", "security",
    "tableau", "power bi", "powerbi", "looker", "excel",
    "agile", "scrum", "jira", "confluence",
    "sap", "oracle", "salesforce", "servicenow",
    "numpy", "pandas", "scipy", "matplotlib", "seaborn", "plotly",
    "hibernate", "jdbc", "jpa", "struts", "jee",
    "asp.net", ".net", "c#", "dotnet",
    "blockchain", "solidity", "web3", "smart contract",
    "api", "soap", "json", "xml", "protobuf",
}
_SINGLE_WORD_SKILLS.update(s for s in _EXTRA_SKILLS if len(s.split()) == 1)
_MULTI_WORD_SKILLS.update(s for s in _EXTRA_SKILLS if len(s.split()) > 1)
for s in _EXTRA_SKILLS:
    norm = s.strip().lower()
    if len(norm.split()) > 1 and norm not in _MULTI_WORD_MAP:
        _MULTI_WORD_MAP[norm] = norm

_MULTI_WORD_SKILLS_SORTED = sorted(_MULTI_WORD_SKILLS, key=lambda x: -len(x.split()))

_SENIORITY_LEVELS = {"junior": "junior", "jr": "junior", "mid": "mid",
                     "senior": "senior", "sr": "senior", "lead": "lead",
                     "principal": "principal", "staff": "staff",
                     "head": "head", "vp": "vp", "director": "director",
                     "c-level": "c-level", "cto": "c-level", "ceo": "c-level",
                     "chief": "c-level", "architect": "architect",
                     "manager": "manager", "head of": "head"}

_NON_SKILL_WORDS: frozenset = frozenset({
    "find", "search", "look", "looking", "need", "wanted", "hiring",
    "and", "the", "a", "an", "in", "at", "on", "to", "of", "is", "are",
    "for", "with", "from", "by", "as", "or", "but", "not", "be",
    "i", "we", "you", "he", "she", "it", "they", "me", "my", "our",
    "senior", "junior", "lead", "head", "principal", "staff",
    "developer", "engineer", "manager", "architect", "analyst",
    "experience", "role", "position", "job", "opening", "hire",
    "dev", "sr", "jr", "intern", "fresher",
    "year", "years", "yr", "yrs", "exp", "experience",
    "building", "designing", "managing", "leading", "working",
    "team", "teams", "people", "product", "projects", "company",
    "good", "strong", "hands-on", "handson", "expert",
    "preferred", "required", "must", "should", "ability",
    "proven", "track", "record", "etc", "including",
    "platform", "service", "system", "application",
    "engineering", "technology", "technologies",
})

_EXPERIENCE_RE = re.compile(
    r'(?P<min>\d+)\+?\s*(?:years?|yrs?|yr)\s*(?:of\s*)?(?:experience)?',
    re.IGNORECASE,
)


def detect_multi_word_skills(text: str) -> list[str]:
    lower = text.lower()
    found: list[str] = []
    start = 0
    while start < len(lower):
        matched = 0
        for phrase in _MULTI_WORD_SKILLS_SORTED:
            n = len(phrase)
            segment = lower[start:start + len(phrase)]
            if segment == phrase:
                pre_ok = start == 0 or not lower[start - 1].isalnum()
                post_ok = (start + len(phrase) >= len(lower)
                           or not lower[start + len(phrase)].isalnum())
                if pre_ok and post_ok:
                    found.append(_MULTI_WORD_MAP.get(phrase, phrase))
                    start += len(phrase)
                    matched = len(phrase)
                    break
        if matched == 0:
            start += 1
    return found


def parse_query(text: str) -> ParsedQuery:
    original = text.strip()
    lower = original.lower()

    multi_skills = detect_multi_word_skills(lower)
    remaining = lower
    for phrase in sorted(multi_skills, key=len, reverse=True):
        remaining = remaining.replace(phrase.lower(), "", 1)
    remaining = re.sub(r'\s+', ' ', remaining).strip()

    seniority: str | None = None
    for label, level in _SENIORITY_LEVELS.items():
        if label in remaining.split():
            seniority = level
            break
        if label == "head of" and "head of" in remaining:
            seniority = "head"
            break

    exp_min: int | None = None
    exp_max: int | None = None
    exp_match = _EXPERIENCE_RE.search(remaining)
    if exp_match:
        exp_min = int(exp_match.group("min"))
        range_match = re.search(
            rf'{exp_match.group("min")}\s*-\s*(\d+)\s*(?:years?|yrs?)',
            remaining,
        )
        if range_match:
            exp_max = int(range_match.group(1))

    city: str | None = None
    for c in sorted(INDIAN_CITIES, key=len, reverse=True):
        if c.lower() in remaining:
            city = c
            remaining = remaining.replace(c.lower(), "")
            break

    words = remaining.split()
    skill_words: list[str] = []
    for w in words:
        w_clean = w.strip(",.!?;:'\"()[]{}")
        if len(w_clean) <= 1:
            continue
        if w_clean in _NON_SKILL_WORDS:
            continue
        if w_clean in _SINGLE_WORD_SKILLS:
            skill_words.append(w_clean)

    seen: set[str] = set()
    all_skills: list[str] = []
    for s in multi_skills + skill_words:
        s_norm = s.strip().lower()
        if s_norm not in seen:
            seen.add(s_norm)
            all_skills.append(s)

    required_skills = [RequiredSkill(name=s, importance=SkillImportance.REQUIRED)
                       for s in all_skills[:8]]
    preferred_skills = [PreferredSkill(name=s, importance=SkillImportance.PREFERRED)
                        for s in all_skills[8:]]

    if not all_skills:
        required_skills = [RequiredSkill(name=original,
                                         importance=SkillImportance.REQUIRED)]

    exp_req = ExperienceRequirements(
        min_years=float(exp_min) if exp_min else None,
        max_years=float(exp_max) if exp_max else None,
    )

    loc = LocationRequirements(city=city or "", remote_ok="remote" in lower)

    return ParsedQuery(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        experience=exp_req,
        location=loc,
        filters=QueryFilters(),
        original_query=original,
    )


def expand_with_aliases(text: str) -> list[str]:
    lower = text.lower()
    expansions: list[str] = [text]

    detected_multi = detect_multi_word_skills(lower)
    detected_multi_norm = set(s.lower() for s in detected_multi)

    words = lower.split()
    for i, w in enumerate(words):
        if w in _NON_SKILL_WORDS:
            continue
        for skill, aliases in SKILL_ALIASES.items():
            all_forms = [skill] + aliases
            if w in all_forms or skill == w or w in aliases:
                for form in all_forms:
                    if form != w and form not in detected_multi_norm:
                        alt_words = list(words)
                        alt_words[i] = form
                        expansions.append(" ".join(alt_words))
                break

    return expansions
