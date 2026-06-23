"""
Behavioral signal scoring + career trajectory analysis + honeypot detection.

Extracts signal from Redrob platform data to differentiate candidates beyond
skill matching. Models real recruiter-style evaluation dimensions.
"""
from __future__ import annotations

import logging
import statistics
from datetime import UTC, datetime

from src.core.models import Profile, Signals

logger = logging.getLogger(__name__)

# ── Honeypot Detection ──────────────────────────────────────────────


def detect_honeypot(profile: Profile) -> str | None:
    """Check if a profile has impossible attributes.

    Returns a reason string if honeypot detected, None if legitimate.
    """
    reasons = []

    # 1. Time-travel: Experience at company before it could exist
    if profile.experience:
        company_founded = {
            "mindtree": 1999,
            "dunder mifflin": 2000,
            "hooli": 2005,
            "acme corp": 2000,
            "globex inc": 1990,
            "pied piper": 2014,
            "infosys": 1981,
            "tcs": 1968,
            "wipro": 1945,
            "cognizant": 1994,
            "tech mahindra": 1986,
            "hcl": 1976,
            "l&t infotech": 1997,
            "mphasis": 1992,
            "oracle": 1977,
            "microsoft": 1975,
            "amazon": 1994,
            "google": 1998,
            "swiggy": 2014,
            "zomato": 2008,
            "razorpay": 2013,
            "ola": 2010,
            "cred": 2018,
            "byju's": 2011,
            "flipkart": 2007,
            "ola electric": 2017,
            "zepto": 2021,
            "nykaa": 2012,
        }
        for job in profile.experience:
            if not job.start_date or not job.company:
                continue
            try:
                start_year = int(str(job.start_date).split("-")[0])
                company_lower = job.company.strip().lower()
                founding = company_founded.get(company_lower)
                if founding and start_year < founding:
                    reasons.append(
                        f"Start year {start_year} before {job.company} founded ({founding})"
                    )
            except (ValueError, IndexError):
                pass

    # 2. Too many skills for experience (unrealistic breadth)
    #    Relaxed: >5 skills/year (a real senior dev could have 4-5 strong skills per year)
    if profile.skills and profile.professional and profile.professional.total_experience_years:
        exp_years = profile.professional.total_experience_years
        if exp_years > 0 and len(profile.skills) / exp_years > 5:
            rate = len(profile.skills) / exp_years
            reasons.append(f"{len(profile.skills)} skills in {exp_years:.0f}y ({rate:.1f}/year)")

    # 3. Expert in 5+ skills with 0 years used
    expert_zero_years = 0
    for skill in profile.skills:
        if (skill.proficiency and "expert" in str(skill.proficiency).lower()
                and (skill.years_used is None or skill.years_used == 0)):
            expert_zero_years += 1
    if expert_zero_years >= 5:
        reasons.append(f"{expert_zero_years} expert skills with 0 years used")

    # 4. Career gap > 5 years with no explanation
    if len(profile.experience) >= 2:
        sorted_exp = sorted(
            [e for e in profile.experience if e.end_date and e.start_date],
            key=lambda e: str(e.start_date or ""),
        )
        for i in range(len(sorted_exp) - 1):
            try:
                curr_end = str(sorted_exp[i].end_date or "")
                next_start = str(sorted_exp[i + 1].start_date or "")
                if curr_end and next_start:
                    end_ym = curr_end.split("-")[:2]
                    start_ym = next_start.split("-")[:2]
                    gap_months = (
                        (int(start_ym[0]) - int(end_ym[0])) * 12
                        + (int(start_ym[1]) - int(end_ym[1]))
                    )
                    if gap_months > 60:
                        reasons.append(f"Gap of {gap_months // 12}y between jobs")
            except (ValueError, IndexError):
                pass

    return "; ".join(reasons) if reasons else None


# ── Career Trajectory Scoring ───────────────────────────────────────


def compute_career_trajectory(profile: Profile) -> float:
    """Score career trajectory quality (0-1).

    Rewards: career progression, stability, company quality.
    Penalizes: job hopping, pure consulting, no progression.
    """
    experiences = profile.experience
    if not experiences:
        return 0.3  # Neutral for single-job profiles

    scores = []

    def _calc_months(start_date: str | None, end_date: str | None) -> int | None:
        """Calculate months between two dates."""
        if not start_date or not end_date:
            return None
        try:
            start_parts = str(start_date).split("-")
            end_parts = str(end_date).split("-")
            if len(start_parts) < 2 or len(end_parts) < 2:
                return None
            return (int(end_parts[0]) - int(start_parts[0])) * 12 \
                + (int(end_parts[1]) - int(start_parts[1]))
        except (ValueError, IndexError):
            return None

    # 1. Job hopping penalty: avg tenure < 18 months at 3+ jobs
    tenures = []
    for job in experiences:
        months = _calc_months(job.start_date, job.end_date)
        if months is not None and months > 0:
            tenures.append(months)

    if tenures and len(experiences) >= 3:
        avg_tenure = statistics.mean(tenures)
        if avg_tenure < 18:
            scores.append(0.3)  # Job hopper
        elif avg_tenure < 36:
            scores.append(0.6)  # Moderate stability
        else:
            scores.append(0.9)  # Strong stability
    elif tenures:
        avg_tenure = statistics.mean(tenures)
        if avg_tenure >= 36:
            scores.append(0.8)
        elif avg_tenure >= 18:
            scores.append(0.7)
        else:
            scores.append(0.5)
    else:
        scores.append(0.5)

    # 2. Consulting detection: Check for consulting/staffing companies
    consulting_keywords = [
        "consulting", "staffing", "contract", "temp", "freelance",
        "randstad", "adecco", "teamlease", "manpower", "kelly services",
    ]
    consulting_jobs = 0
    for job in experiences:
        company_lower = (job.company or "").lower()
        title_lower = (job.title or "").lower()
        if any(kw in company_lower for kw in consulting_keywords[:3]):
            consulting_jobs += 1
        elif any(kw in title_lower for kw in consulting_keywords):
            consulting_jobs += 1

    if consulting_jobs == len(experiences):
        scores.append(0.25)  # Full consulting career
    elif consulting_jobs > len(experiences) / 2:
        scores.append(0.5)   # Majority consulting
    else:
        scores.append(0.85)  # Not consulting

    # 3. Career progression (title trajectory)
    if len(experiences) >= 2:
        titles = [str(job.title or "") for job in experiences]
        # Check for progression keywords
        progression_signals = 0
        for i in range(len(titles) - 1):
            curr = titles[i].lower()
            prev = titles[i + 1].lower()
            # Current role is more senior
            senior_kw = ["senior", "lead", "head", "principal", "staff",
                         "architect", "manager", "director", "vp", "chief"]
            if any(kw in curr for kw in senior_kw):
                if not any(kw in prev for kw in senior_kw):
                    progression_signals += 1
        progression_rate = progression_signals / max(1, len(titles) - 1)
        scores.append(0.3 + 0.6 * progression_rate)
    else:
        scores.append(0.6)  # Neutral for single-job

    return statistics.mean(scores)


# ── Behavioral Signal Scoring ───────────────────────────────────────


def compute_behavioral_score(signals: Signals) -> float:
    """Score platform behavioral signals (0-1).

    Uses: response rate, saved count, completeness, verification,
          engagement, github activity.
    """
    components = []

    # 1. Recruiter response rate (shows engagement with opportunities)
    if signals.recruiter_response_rate is not None and signals.recruiter_response_rate >= 0:
        components.append(min(1.0, signals.recruiter_response_rate * 1.2))
    else:
        components.append(0.3)  # Unknown — neutral

    # 2. Saved by recruiters (market demand signal)
    if signals.saved_by_recruiters_30d is not None:
        saved_norm = min(1.0, signals.saved_by_recruiters_30d / 15.0)
        components.append(saved_norm)
    else:
        components.append(0.3)

    # 3. Profile completeness (candidate quality)
    if signals.profile_completeness_score is not None:
        components.append(signals.profile_completeness_score / 100.0)
    else:
        components.append(0.2)

    # 4. Verification (authenticity)
    verified_count = 0
    verified_total = 0
    if signals.verified_email is not None:
        verified_total += 1
        if signals.verified_email:
            verified_count += 1
    if signals.verified_phone is not None:
        verified_total += 1
        if signals.verified_phone:
            verified_count += 1
    if signals.linkedin_connected is not None:
        verified_total += 1
        if signals.linkedin_connected:
            verified_count += 1
    if verified_total > 0:
        components.append(verified_count / verified_total)
    else:
        components.append(0.3)

    # 5. GitHub activity (tech signal)
    if signals.github_activity_score is not None:
        gh = max(0, signals.github_activity_score)
        components.append(min(1.0, gh / 30.0))
    else:
        components.append(0.2)

    # 6. Open to work / willing to relocate (availability)
    if signals.open_to_work is not None and signals.open_to_work:
        components.append(0.9)
    elif signals.open_to_work is not None:
        components.append(0.5)
    else:
        components.append(0.4)

    if signals.willing_to_relocate is not None and signals.willing_to_relocate:
        components.append(0.9)
    elif signals.willing_to_relocate is not None:
        components.append(0.5)
    else:
        components.append(0.4)

    # 7. Interview completion rate (engaged candidate)
    if signals.interview_completion_rate is not None and signals.interview_completion_rate >= 0:
        components.append(signals.interview_completion_rate)
    else:
        components.append(0.5)

    # 8. Recency (active in last 30 days)
    if signals.last_active_date:
        try:
            last_active = datetime.strptime(signals.last_active_date, "%Y-%m-%d")
            days_since = (datetime.now(UTC) - last_active.replace(tzinfo=UTC)).days
            recency = max(0, min(1.0, 1.0 - days_since / 365.0))
            components.append(recency)
        except ValueError:
            components.append(0.3)
    else:
        components.append(0.2)

    return statistics.mean(components) if components else 0.3


# ── Skill Proficiency Scoring ───────────────────────────────────────


def compute_skill_proficiency(profile: Profile, required_skill_names: list[str]) -> float:
    """Score skill depth based on proficiency level, endorsements, and verified platform tests."""
    if not required_skill_names or not profile.skills:
        return 0.0

    skill_score = []
    required_lower = {s.lower() for s in required_skill_names}
    assessment_scores = profile.signals.skill_assessment_scores or {}

    for skill in profile.skills:
        if skill.name.lower() in required_lower:
            # Base score
            base = 0.5

            # Proficiency bonus
            if skill.proficiency:
                prof_str = str(skill.proficiency).lower()
                if "expert" in prof_str:
                    base = 1.0
                elif "advanced" in prof_str:
                    base = 0.85
                elif "intermediate" in prof_str:
                    base = 0.7
                elif "beginner" in prof_str:
                    base = 0.5

            # Verified platform assessment — strongest signal
            if skill.name in assessment_scores:
                test_score = assessment_scores[skill.name] / 100.0
                base = max(base, test_score)

            # Years used bonus
            if skill.years_used:
                base = min(1.0, base + skill.years_used * 0.03)

            # Endorsements bonus (via confidence field)
            if skill.confidence:
                base = min(1.0, base + skill.confidence * 0.1)

            skill_score.append(base)

    return statistics.mean(skill_score) if skill_score else 0.0
