"""Explainable ranking: why this candidate, with quoted evidence.

For one (query, candidate, scores) triple this module answers the three questions a
recruiter asks before trusting a ranked list:

1. Where did the score come from?  -> ``contributions``: each dimension's weight x score,
   expressed as points of the final score (they sum to ``overall``), plus the role/title
   multiplier the executor applies.
2. Does the candidate actually have what I asked for?  -> ``requirements``: every skill,
   location, experience band and role in the query, checked against the profile, with the
   *strength* of the evidence and the exact text spans that support it:

      verified     - a platform skill-assessment score exists for it
      structured   - listed in the skills section (proficiency / years shown)
      demonstrated - appears in the title or description of an actual role
      mentioned    - only appears in free text (summary/headline): a claim, not proof

3. What could go wrong?  -> ``risks`` (integrity flags, long notice, unresponsiveness,
   job-hopping, claimed-but-unproven skills) and ``strengths``.

Everything is deterministic and model-free, so explanations are reproducible and cheap
enough to attach to every result.
"""

from __future__ import annotations

import re
import statistics
from datetime import UTC, datetime
from typing import Literal, cast

from src.core.models import (
    EvidenceSpan,
    Explanation,
    ParsedQuery,
    Profile,
    RequirementCheck,
    ScoreContribution,
)
from src.matching.behavioral_scorer import detect_honeypot
from src.matching.experience_matcher import ExperienceMatcher
from src.matching.skill_matcher import SKILL_ALIASES

Status = Literal["met", "partial", "missing"]
Verdict = Literal["strong", "good", "potential", "weak"]

DIMENSION_LABELS: dict[str, str] = {
    "skill_match": "Required skills",
    "semantic_similarity": "Semantic fit",
    "cross_encoder_score": "Relevance rank",
    "behavioral_score": "Engagement signals",
    "keyword_match": "Keyword overlap",
    "experience_match": "Experience",
    "career_trajectory_score": "Career trajectory",
    "skill_proficiency_score": "Skill depth",
    "location_match": "Location",
    "education_match": "Education",
}

_STRENGTH_ORDER = {"verified": 0, "structured": 1, "demonstrated": 2, "mentioned": 3}
_CITY_FOLD = {"bengaluru": "bangalore", "gurugram": "gurgaon", "bombay": "mumbai",
              "new delhi": "delhi", "madras": "chennai", "calcutta": "kolkata"}
_SNIPPET_RADIUS = 70
_MAX_SPANS_PER_TERM = 3


# ── helpers ─────────────────────────────────────────────────────────


def _aliases(term: str, for_prose: bool = False) -> list[str]:
    """The term plus its known aliases, longest first. For free-text search
    (``for_prose``) names shorter than 3 characters are dropped, the term itself
    included: ``go``/``r``/``c``/``ml`` match too much prose ("go-live", "R&D")."""
    t = term.strip().lower()
    names = {t}
    names.update(SKILL_ALIASES.get(t, []))
    for canonical, aliases in SKILL_ALIASES.items():
        if t in aliases:
            names.add(canonical)
            names.update(aliases)
    keep = {n for n in names if n and (len(n) >= 3 or not for_prose)}
    return sorted(keep, key=len, reverse=True)


def _pattern(names: list[str]) -> re.Pattern[str]:
    alts = "|".join(re.escape(n) for n in names)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alts})(?![A-Za-z0-9])", re.IGNORECASE)


def _snippet(text: str, start: int, end: int) -> tuple[str, int, int]:
    """Window around [start, end), snapped to word boundaries, with ellipses."""
    lo = max(0, start - _SNIPPET_RADIUS)
    hi = min(len(text), end + _SNIPPET_RADIUS)
    if lo > 0:
        space = text.find(" ", lo, start)
        lo = space + 1 if space != -1 else lo
    if hi < len(text):
        space = text.rfind(" ", end, hi)
        hi = space if space != -1 else hi
    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(text) else ""
    body = " ".join(text[lo:hi].split("\n"))
    snippet = f"{prefix}{body}{suffix}"
    s = len(prefix) + (start - lo)
    return snippet, s, s + (end - start)


def _ym(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = str(value).split("-")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 1
    except (ValueError, IndexError):
        return None


def _months(start: str | None, end: str | None) -> int | None:
    a, b = _ym(start), _ym(end)
    if a is None or b is None:
        return None
    m = (b[0] - a[0]) * 12 + (b[1] - a[1])
    return m if m >= 0 else None


def _period(start: str | None, end: str | None, current: bool) -> str:
    a = _ym(start)
    b = _ym(end)
    left = str(a[0]) if a else "?"
    right = "present" if current or b is None else str(b[0])
    return f"{left}–{right}"


def _fold_city(city: str | None) -> str:
    c = (city or "").split(",")[0].strip().lower()
    return _CITY_FOLD.get(c, c)


# ── evidence search ─────────────────────────────────────────────────


def find_evidence(term: str, profile: Profile) -> list[EvidenceSpan]:
    """All evidence for one skill-like term, strongest first (one span per source item)."""
    # Skills section / assessments: exact names, short aliases allowed (JS, ML, Go).
    pat = _pattern(_aliases(term))
    prose_names = _aliases(term, for_prose=True)
    prose = _pattern(prose_names) if prose_names else None
    spans: list[EvidenceSpan] = []

    for skill_name, score in (profile.signals.skill_assessment_scores or {}).items():
        if pat.fullmatch(skill_name.strip()):
            text = f"{skill_name} — platform assessment {score:.0f}/100"
            spans.append(EvidenceSpan(term=term, source="assessment", label="Skill assessment",
                                      text=text, start=0, end=len(skill_name),
                                      strength="verified"))

    for skill in profile.skills:
        if pat.fullmatch(skill.name.strip()):
            bits = [skill.name]
            if skill.proficiency:
                bits.append(str(skill.proficiency.value if hasattr(skill.proficiency, "value")
                                else skill.proficiency))
            if skill.years_used:
                bits.append(f"{skill.years_used:g} yrs")
            text = " · ".join(bits)
            spans.append(EvidenceSpan(term=term, source="skills", label="Skills section",
                                      text=text, start=0, end=len(skill.name),
                                      strength="structured"))
            break

    for job in profile.experience:
        period = _period(job.start_date, job.end_date, job.is_current)
        label = f"{job.title} @ {job.company} ({period})"
        for field_text in [job.title, job.description, *job.highlights]:
            if not field_text:
                continue
            m = prose.search(field_text) if prose else None
            if m:
                text, s, e = _snippet(field_text, m.start(), m.end())
                spans.append(EvidenceSpan(term=term, source="experience", label=label,
                                          text=text, start=s, end=e, strength="demonstrated"))
                break

    raw = profile.raw_text or ""
    summary_at = raw.find("Summary:")
    summary = raw[summary_at + len("Summary:"):].strip() if summary_at != -1 else raw
    m = prose.search(summary) if prose else None
    if m:
        text, s, e = _snippet(summary, m.start(), m.end())
        spans.append(EvidenceSpan(term=term, source="summary", label="Profile summary",
                                  text=text, start=s, end=e, strength="mentioned"))

    spans.sort(key=lambda sp: _STRENGTH_ORDER[sp.strength])
    return spans


def _skill_check(term: str, profile: Profile, required: bool) -> RequirementCheck:
    spans = find_evidence(term, profile)
    kind_word = "Required" if required else "Nice-to-have"
    if not spans:
        return RequirementCheck(requirement=term, kind="skill", status="missing",
                                detail=f"{kind_word}: no mention anywhere in the profile")
    best = spans[0].strength
    detail = {
        "verified": "backed by a platform assessment score",
        "structured": "listed in the skills section",
        "demonstrated": "used in a past or current role",
        "mentioned": "only claimed in the summary, not listed or shown in any role",
    }[best]
    status = "partial" if best == "mentioned" else "met"
    return RequirementCheck(requirement=term, kind="skill", status=cast(Status, status),
                            strength=best, detail=f"{kind_word}: {detail}",
                            evidence=spans[:_MAX_SPANS_PER_TERM])


# ── trajectory ──────────────────────────────────────────────────────


def trajectory_facts(profile: Profile, include_current_tenure: bool,
                     as_of: str | None = None) -> dict[str, float | int | str | None]:
    today = as_of or datetime.now(UTC).date().isoformat()
    completed = [m for j in profile.experience
                 if (m := _months(j.start_date, j.end_date)) is not None and m > 0]
    current = None
    for j in profile.experience:
        if j.is_current and j.start_date:
            current = _months(j.start_date, today)
            break
    counted = completed + ([current] if include_current_tenure and current else [])
    return {
        "mode": "include_current" if include_current_tenure else "completed_roles",
        "roles": len(profile.experience),
        "avg_tenure_months": round(statistics.mean(counted), 1) if counted else None,
        "current_tenure_months": current,
    }


# ── main entry point ────────────────────────────────────────────────


def explain_candidate(
    profile: Profile,
    parsed: ParsedQuery,
    scores: dict[str, float | None],
    overall: float,
    weights: dict[str, float],
    include_current_tenure: bool = False,
    as_of: str | None = None,
    credit_work_history: bool = False,
) -> Explanation:
    """Build the explanation. ``scores``/``weights`` are per dimension, as used by
    CandidateScorer; ``overall`` is the final (title-adjusted) score."""
    # 1. Contributions. The executor multiplies the weighted mean by a role/title factor;
    # recover it so the points add up to the score the recruiter sees.
    present = {d: s for d, s in scores.items() if s is not None and weights.get(d, 0) > 0}
    total_w = sum(weights[d] for d in present)
    mean = sum(weights[d] * s for d, s in present.items()) / total_w if total_w else 0.0
    title_factor = max(0.0, min(1.0, overall / mean)) if mean > 0 else 1.0
    contributions = [
        ScoreContribution(
            dimension=d, label=DIMENSION_LABELS.get(d, d.replace("_", " ").title()),
            score=round(s, 4), weight=round(weights[d] / total_w, 4),
            points=round(weights[d] * s / total_w * title_factor, 4),
        )
        for d, s in present.items()
    ]
    contributions.sort(key=lambda c: -c.points)

    # 2. Requirements.
    requirements: list[RequirementCheck] = []
    seen: set[str] = set()
    for req, is_required in ([(r.name, True) for r in parsed.required_skills]
                             + [(p.name, False) for p in parsed.preferred_skills]):
        key = req.strip().lower()
        if not key or key in seen or key == (parsed.original_query or "").strip().lower():
            continue
        seen.add(key)
        requirements.append(_skill_check(req, profile, is_required))

    query = parsed.original_query or ""
    matcher = ExperienceMatcher()
    role = matcher._extract_role(query)
    if role and role not in ("engineer", "developer"):
        current_title = profile.professional.current_title or ""
        fit = matcher._score_title_compatibility(role, current_title)
        status = "met" if fit >= 0.9 else "partial" if fit >= 0.5 else "missing"
        ev: list[EvidenceSpan] = []
        if current_title:
            ev.append(EvidenceSpan(term=role, source="headline", label="Current title",
                                   text=current_title, start=0, end=len(current_title),
                                   strength="structured"))
        requirements.insert(0, RequirementCheck(
            requirement=role.title(), kind="role", status=cast(Status, status),
            strength="structured" if current_title else None,
            detail=f"Current title is {current_title or 'unknown'} "
                   f"({'same' if fit >= 0.9 else 'adjacent' if fit >= 0.5 else 'different'} track)",
            evidence=ev))

    years = profile.professional.total_experience_years
    lo, hi = parsed.experience.min_years, parsed.experience.max_years
    if lo is not None or hi is not None:
        band = (f"{lo:g}–{hi:g} yrs" if lo is not None and hi is not None
                else f"{lo:g}+ yrs" if lo is not None else f"up to {hi:g} yrs")
        if years is None:
            status, detail = "missing", "Total experience not stated"
        else:
            below = lo is not None and years < lo
            above = hi is not None and years > hi
            gap = (lo - years) if below and lo is not None else (
                years - hi if above and hi is not None else 0.0)
            status = "met" if not (below or above) else "partial" if gap <= 1 else "missing"
            detail = f"{years:g} years total"
        requirements.append(RequirementCheck(requirement=band, kind="experience",
                                             status=cast(Status, status), detail=detail))

    if parsed.location.city:
        want = _fold_city(parsed.location.city)
        have_raw = profile.personal.location.city if profile.personal else None
        have = _fold_city(have_raw)
        if have and have == want:
            status, detail = "met", f"Based in {have_raw}"
        elif profile.signals.willing_to_relocate:
            status, detail = "partial", f"Based in {have_raw or 'unknown'}, willing to relocate"
        else:
            status, detail = "missing", f"Based in {have_raw or 'unknown'}"
        requirements.append(RequirementCheck(requirement=parsed.location.city.title(),
                                             kind="location", status=cast(Status, status),
                                             detail=detail))

    # 3. Strengths and risks.
    sig = profile.signals
    strengths: list[str] = []
    risks: list[str] = []
    verified_skills = [r.requirement for r in requirements
                       if r.kind == "skill" and r.strength == "verified"]
    if verified_skills:
        strengths.append(f"Assessment-verified: {', '.join(verified_skills)}")
    if sig.recruiter_response_rate is not None and sig.recruiter_response_rate >= 0.7:
        strengths.append(f"Replies to {sig.recruiter_response_rate:.0%} of recruiter messages")
    if sig.open_to_work:
        strengths.append("Open to work")
    if sig.notice_period_days is not None and sig.notice_period_days <= 30:
        strengths.append(f"Can join in {sig.notice_period_days} days")
    if sig.interview_completion_rate is not None and sig.interview_completion_rate >= 0.85:
        strengths.append(f"Completes {sig.interview_completion_rate:.0%} of interviews")
    if sig.github_activity_score is not None and sig.github_activity_score >= 60:
        strengths.append(f"Active on GitHub (activity {sig.github_activity_score:.0f}/100)")
    if sig.verified_email and sig.verified_phone:
        strengths.append("Verified email and phone")

    skill_score = scores.get("skill_match")
    off_list = [r.requirement for r in requirements
                if r.kind == "skill" and r.strength == "demonstrated"]
    if off_list and credit_work_history:
        strengths.append(f"Credited from work history: {', '.join(off_list)}")
    elif off_list and skill_score is not None and skill_score < 1.0:
        strengths.append(
            f"Proven in work history but missing from the skills list: {', '.join(off_list)} "
            "(the skills score only counts listed skills)"
        )

    flag = detect_honeypot(profile)
    if flag:
        risks.append(f"Integrity flag: {flag} (score ×0.15)")
    claimed = [r.requirement for r in requirements
               if r.kind == "skill" and r.strength == "mentioned"]
    if claimed:
        risks.append(f"Claimed but unproven: {', '.join(claimed)} (summary only)")
    missing_req = [r.requirement for r in requirements
                   if r.kind == "skill" and r.status == "missing" and r.detail.startswith("Req")]
    if missing_req:
        risks.append(f"No evidence of required skill{'s' if len(missing_req) > 1 else ''}: "
                     f"{', '.join(missing_req)}")
    if sig.notice_period_days is not None and sig.notice_period_days >= 90:
        risks.append(f"{sig.notice_period_days}-day notice period")
    if sig.recruiter_response_rate is not None and sig.recruiter_response_rate < 0.3:
        risks.append(f"Rarely replies to recruiters ({sig.recruiter_response_rate:.0%})")
    if sig.is_passive:
        risks.append("Passive candidate: not actively looking")
    traj = trajectory_facts(profile, include_current_tenure, as_of)
    avg = traj.get("avg_tenure_months")
    roles = traj.get("roles")
    if isinstance(avg, (int, float)) and isinstance(roles, int) and roles >= 3 and avg < 18:
        risks.append(f"Short stints: {avg:.0f}-month average tenure over {roles} roles")

    # 4. Verdict + one-line headline.
    # Thresholds on the 0-100 score as displayed, so "70/100" never reads "good".
    shown = round(overall * 100)
    verdict = ("strong" if shown >= 70 else "good" if shown >= 50
               else "potential" if shown >= 30 else "weak")
    skill_reqs = [r for r in requirements if r.kind == "skill"]
    met = sum(r.status == "met" for r in skill_reqs)
    title = profile.professional.current_title or "Candidate"
    parts = [f"{title}"]
    if profile.professional.current_company:
        parts.append(f"at {profile.professional.current_company}")
    head = " ".join(parts)
    if skill_reqs:
        head += f" · proves {met}/{len(skill_reqs)} asked skills"
    if contributions:
        head += f" · driven by {contributions[0].label.lower()}"
    if flag:
        head += " · integrity flag"

    return Explanation(
        profile_id=profile.profile_id, verdict=cast(Verdict, verdict), headline=head,
        overall=round(overall, 4), title_factor=round(title_factor, 4),
        contributions=contributions, requirements=requirements,
        strengths=strengths, risks=risks, trajectory=traj,
    )
