from __future__ import annotations

import math

from src.core.models import MatchResult, MatchScores, Profile, Rationale, SearchResultItem


def _bullet_years(years: float | None) -> str:
    return f" \u2022 {years:.0f}yrs exp" if years else ""


MATCH_COLORS = {
    "strong_match": "#10b981",
    "good_match": "#3b82f6",
    "potential_match": "#f59e0b",
    "weak_match": "#ef4444",
}


def _score_bar(label: str, value: float, color: str = "#3b82f6") -> str:
    pct = max(0, min(100, int(value * 100)))
    return f"""
    <div style="display:flex;align-items:center;margin:2px 0;gap:6px;">
        <span style="font-size:11px;color:#6b7280;width:85px;text-align:right;">{label}</span>
        <div style="flex:1;height:8px;background:#e5e7eb;border-radius:4px;">
            <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>
        </div>
        <span style="font-size:11px;color:#374151;width:30px;text-align:right;">{pct}%</span>
    </div>"""


def _score_badge(value: float) -> str:
    pct = int(value * 100)
    if pct >= 70:
        color = "#10b981"
        label = "strong"
    elif pct >= 50:
        color = "#3b82f6"
        label = "good"
    elif pct >= 30:
        color = "#f59e0b"
        label = "potential"
    else:
        color = "#ef4444"
        label = "weak"
    return f"""
    <div class="score-badge score-{label}"
         style="background:{color}15;color:{color};border:1px solid {color}30;">
        <span style="font-size:20px;font-weight:700;">{pct}</span>
        <span style="font-size:11px;opacity:0.8;">%</span>
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
    </div>"""


def create_candidate_card(item: SearchResultItem) -> str:
    s = item.scores if item.scores is not None else MatchScores()
    bars = ""
    bars += _score_bar("Overall", s.overall, "#6366f1")
    bars += _score_bar("Skill", s.skill_match, "#10b981")
    bars += _score_bar("Experience", s.experience_match, "#3b82f6")
    bars += _score_bar("Semantic", s.semantic_similarity, "#8b5cf6")
    bars += _score_bar("Keyword", s.keyword_match, "#f59e0b")
    if s.education_match is not None:
        bars += _score_bar("Education", s.education_match, "#ec4899")
    if s.cross_encoder_score is not None:
        bars += _score_bar("AI Rerank", s.cross_encoder_score, "#06b6d4")

    skills_html = "".join(
        f'<span class="skill-chip matched">{skill}</span>' for skill in item.matched_skills[:8]
    )
    missing_html = "".join(
        f'<span class="skill-chip missing">{skill}</span>' for skill in item.missing_skills[:5]
    )

    return f"""
    <div class="candidate-card">
        <div style="display:flex;justify-content:space-between;align-items:start;gap:16px;">
            <div style="flex:1;">
                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                    <strong style="font-size:18px;">{item.name}</strong>
                    <span style="font-size:12px;color:#9ca3af;">#{item.rank}</span>
                </div>
                <div style="color:#6b7280;margin-top:4px;">
                    {item.current_title or 'N/A'}
                    {(
            f' <span style="color:#9ca3af;">at</span> '
            f'<strong>{item.current_company}</strong>'
            if item.current_company else ""
        )}
                </div>
                <div style="color:#9ca3af;font-size:13px;margin-top:2px;">
                    {item.location or 'Location N/A'}
                    {_bullet_years(item.experience_years)}
                </div>
            </div>
            {_score_badge(s.overall)}
        </div>
        <div style="margin-top:12px;">
            <div style="font-size:13px;color:#374151;margin-bottom:4px;">
                Score Breakdown
                <span style="color:#9ca3af;font-size:11px;">(confidence: {s.confidence:.0%})</span>
            </div>
            {bars}
        </div>
        <div style="margin-top:12px;display:flex;flex-wrap:wrap;gap:4px;">
            {skills_html}
            {missing_html}
        </div>
    </div>
    """


def create_score_radar_chart(scores: dict) -> str:
    dims = ["Skill", "Experience", "Semantic", "Keyword", "Confidence"]
    dim_keys = [
        "skill_match", "experience_match",
        "semantic_similarity", "keyword_match", "confidence",
    ]
    values = [int(scores.get(k, 0) * 100) for k in dim_keys]

    cx, cy, r = 100, 100, 80
    angles = [math.radians(90 - i * 72) for i in range(5)]
    outer_points = " ".join(
        f"{cx + r * math.cos(a):.1f},{cy - r * math.sin(a):.1f}" for a in angles
    )
    data_points = " ".join(
        f"{cx + v * r / 100 * math.cos(a):.1f},{cy - v * r / 100 * math.sin(a):.1f}"
        for v, a in zip(values, angles)
    )

    labels = "".join(
        f'<text x="{cx + (r + 18) * math.cos(a):.1f}" y="{cy - (r + 18) * math.sin(a):.1f}" '
        f'font-size="11" text-anchor="middle" fill="#374151">{dim}</text>'
        for dim, a in zip(dims, angles)
    )
    grid_lines = "".join(
        f'<polygon points="{cx + r * frac * math.cos(a):.1f},{cy - r * frac * math.sin(a):.1f} '
        for frac in [0.25, 0.5, 0.75]
        for a in angles
    )
    grid_lines = ""
    for frac in [0.25, 0.5, 0.75]:
        pts = " ".join(
            f"{cx + r * frac * math.cos(a):.1f},{cy - r * frac * math.sin(a):.1f}" for a in angles
        )
        grid_lines += f'<polygon points="{pts}" fill="none" stroke="#e5e7eb" stroke-width="1"/>'

    return f"""
    <svg width="220" height="220" viewBox="0 0 220 220">
        <polygon points="{outer_points}" fill="#f3f4f6" stroke="#d1d5db" stroke-width="1"/>
        {grid_lines}
        <polygon points="{data_points}" fill="#3b82f640" stroke="#3b82f6" stroke-width="2"/>
        <circle cx="{cx}" cy="{cy}" r="2" fill="#3b82f6"/>
        {labels}
    </svg>
    """


def create_skill_match_table(rationale: Rationale) -> str:
    rows = ""
    for sd in rationale.skill_details:
        icon = "\u2705" if sd.found else "\u274c"
        rows += f"<tr><td>{icon}</td><td>{sd.skill}</td><td>{sd.evidence}</td></tr>"

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead>
            <tr style="background:#f9fafb;">
                <th style="padding:8px;text-align:left;">Status</th>
                <th style="padding:8px;text-align:left;">Skill</th>
                <th style="padding:8px;text-align:left;">Evidence</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


def create_analytics_dashboard(results_json: str = "[]") -> str:
    import json

    from src.core.models import MatchResult, MatchScores
    from src.fairness.metrics import (
        compute_all_fairness_metrics,
    )

    # Early return for empty results
    if not results_json or results_json.strip() in ("[]", "", "{}"):
        return (
            "<div style='padding:60px;text-align:center;color:#9ca3af;'>"
            "<p style='font-size:24px;margin-bottom:12px;'>&#128202;</p>"
            "<p style='font-size:18px;'>No results to analyze</p>"
            "<p style='font-size:14px;'>Run a search first to see analytics and fairness metrics.</p>"  # noqa: E501
            "</div>"
        )

    try:
        raw = json.loads(results_json) if results_json else []
    except (json.JSONDecodeError, TypeError):
        raw = []

    total = len(raw)
    listwise_ranked = any(r.get("listwise_ranked", False) for r in raw) if raw else False
    listwise_badge = (
        '<span style="background:#6366f140;color:#6366f1;padding:2px 8px;'
        'border-radius:4px;font-size:12px;">Listwise Ranked</span>'
        if listwise_ranked else ""
    )

    # Build MatchResults + determine metadata from first item
    match_results = []
    for r in raw:
        if isinstance(r, dict):
            scores_dict = r.get("scores", {})
            if not isinstance(scores_dict, dict):
                scores_dict = {}
            match_scores = MatchScores(
                overall=scores_dict.get("overall", r.get("_re_score", 0)),
                semantic_similarity=scores_dict.get("semantic_similarity"),
                keyword_match=scores_dict.get("keyword_match"),
                skill_match=scores_dict.get("skill_match", 0),
                experience_match=scores_dict.get("experience_match", 0),
                location_match=scores_dict.get("location_match"),
                education_match=scores_dict.get("education_match"),
                confidence=scores_dict.get("confidence", 0),
            )
            match_results.append(
                MatchResult(
                    query_id="",
                    profile_id=r.get("profile_id", ""),
                    rank=r.get("rank", 1),
                    name=r.get("name", ""),
                    scores=match_scores,
                    matched_skills=r.get("matched_skills", []),
                    missing_skills=r.get("missing_skills", []),
                )
            )
            if r.get("_listwise_ranked"):
                listwise_ranked = True

    # Compute score distribution
    scores = [m.scores.overall for m in match_results if m.scores.overall > 0]
    if scores:
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        bins = [0] * 10
        for s in scores:
            idx = min(9, int(s * 10))
            bins[idx] += 1
        max_bin = max(bins) if bins else 1
        bar_chart = "".join(
            f'<div style="flex:1;background:{"#6366f1" if i >= 4 else "#93c5fd"};'
            f'height:{max(4, int(b * 120 / max_bin))}px;'
            f'border-radius:4px 4px 0 0;" '
            f'title="{int(i*10)}-{int(i*10+9)}%: {b} candidates"></div>'
            for i, b in enumerate(bins)
        )
    else:
        avg_score = 0
        max_score = 0
        min_score = 0
        bar_chart = '<div style="color:#9ca3af;padding:40px;text-align:center;">No results to analyze</div>'  # noqa: E501

    # Compute fairness metrics
    metric_cards = ""
    if len(match_results) >= 3:
        profiledict = _get_bias_profiles(match_results)
        fairness = compute_all_fairness_metrics(match_results, profiledict)
        dp = fairness.get("demographic_parity", {})
        lang_bias = fairness.get("language_bias", {})

        # Helper
        def _metric_card(label, value, threshold, format_str="{:.3f}"):
            val = value if isinstance(value, (int, float)) else 0
            if val < threshold:
                color = "#10b981"
                status = "No bias detected"
            elif val < threshold * 2:
                color = "#f59e0b"
                status = "Monitor closely"
            else:
                color = "#ef4444"
                status = "Bias detected"
            return f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color};">{format_str.format(val)}</div>
                <div style="font-size:12px;color:#9ca3af;">{status}</div>
            </div>"""

        metric_cards = _metric_card("University Parity", dp.get("university", 1.0), 0.8)
        metric_cards += _metric_card("City Parity", dp.get("city", 1.0), 0.8)
        metric_cards += _metric_card("Language Parity", dp.get("language", 1.0), 0.8)
        metric_cards += _metric_card(
            "Language Avg Rank Diff",
            abs(lang_bias.get("rank_diff", 0)),
            2.0,
            "{:.1f} ranks",
        )

    grid_style = "display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;"
    axis_style = (
        "display:flex;justify-content:space-between;"
        "font-size:11px;color:#9ca3af;margin-top:4px;"
    )

    header_style = (
        "display:flex;justify-content:space-between;"
        "align-items:center;margin-bottom:16px;"
    )
    return f"""
    <div style="padding:20px;">
        <div style="{header_style}">
            <h3 style="margin:0;">Fairness & Bias Metrics</h3>
            <div style="display:flex;gap:8px;align-items:center;">
                {listwise_badge}
                <span style="background:#10b98120;color:#10b981;padding:2px 8px;
                    border-radius:4px;font-size:12px;">PII Anonymized</span>
                <span style="font-size:12px;color:#9ca3af;">
                    {total} candidates analyzed
                </span>
            </div>
        </div>
        <div style="{grid_style}">
            {metric_cards or _empty_metric_card()}
        </div>

        <h3 style="margin:24px 0 16px;">Score Distribution</h3>
        <div style="display:flex;align-items:flex-end;gap:4px;height:130px;">
            {bar_chart}
        </div>
        <div style="{axis_style}">
            <span>0%</span><span>50%</span><span>100%</span>
        </div>
        <div style="display:flex;gap:24px;margin-top:12px;font-size:13px;color:#6b7280;">
            <span>Avg: <strong>{avg_score:.1%}</strong></span>
            <span>Max: <strong>{max_score:.1%}</strong></span>
            <span>Min: <strong>{min_score:.1%}</strong></span>
        </div>

        {_build_distribution_table(match_results)}
    </div>
    """


def _build_distribution_table(match_results: list) -> str:
    """Build a summary table of match categories."""
    strong = sum(1 for m in match_results if m.scores.overall >= 0.8)
    good = sum(1 for m in match_results if 0.6 <= m.scores.overall < 0.8)
    potential = sum(1 for m in match_results if 0.4 <= m.scores.overall < 0.6)
    weak = sum(1 for m in match_results if m.scores.overall < 0.4)
    total = len(match_results) or 1
    return f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px;">
        <div style="background:#10b98115;padding:12px;border-radius:8px;text-align:center;">
            <div style="font-size:24px;font-weight:700;color:#10b981;">{strong}</div>
            <div style="font-size:12px;color:#6b7280;">Strong ({strong*100//total}%)</div>
        </div>
        <div style="background:#3b82f615;padding:12px;border-radius:8px;text-align:center;">
            <div style="font-size:24px;font-weight:700;color:#3b82f6;">{good}</div>
            <div style="font-size:12px;color:#6b7280;">Good ({good*100//total}%)</div>
        </div>
        <div style="background:#f59e0b15;padding:12px;border-radius:8px;text-align:center;">
            <div style="font-size:24px;font-weight:700;color:#f59e0b;">{potential}</div>
            <div style="font-size:12px;color:#6b7280;">Potential ({potential*100//total}%)</div>
        </div>
        <div style="background:#ef444415;padding:12px;border-radius:8px;text-align:center;">
            <div style="font-size:24px;font-weight:700;color:#ef4444;">{weak}</div>
            <div style="font-size:12px;color:#6b7280;">Weak ({weak*100//total}%)</div>
        </div>
    </div>"""


def _empty_metric_card() -> str:
    return (
        '<div class="metric-card">'
        '<div class="metric-label">Run a search to see metrics</div>'
        '</div>'
    )


def _get_bias_profiles(match_results: list[MatchResult]) -> dict[str, Profile]:
    """Build minimal Profile objects for bias detection from match results."""
    from src.core.models import Location, PersonalInfo, Profile, ProfileMetadata
    return {
        m.profile_id: Profile(
            profile_id=m.profile_id,
            personal=PersonalInfo(
                name=m.name or "",
                location=Location(city=m.location),
                languages_spoken=[],
            ),
            metadata=ProfileMetadata(language_detected="en"),
        )
        for m in match_results
    }


def create_rationale_panel(rationale: Rationale | None, profile_summary: str) -> str:
    if rationale is None:
        return ""

    color = MATCH_COLORS.get(rationale.recommendation.value, "#6b7280")
    summary = rationale.summary or "No summary available."

    strengths_html = "".join(
        f"<li>{s}</li>" for s in rationale.strengths[:5]
    )
    gaps_html = "".join(
        f"<li>{g}</li>" for g in rationale.gaps[:5]
    )

    return f"""
    <div class="candidate-card" style="border-left:4px solid {color};">
        <h4>Rationale: {profile_summary}</h4>
        <p style="color:#374151;">{summary}</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px;">
            <div>
                <strong style="color:#10b981;">Strengths</strong>
                <ul style="margin:8px 0;padding-left:20px;font-size:13px;">{strengths_html}</ul>
            </div>
            <div>
                <strong style="color:#ef4444;">Gaps</strong>
                <ul style="margin:8px 0;padding-left:20px;font-size:13px;">{gaps_html}</ul>
            </div>
        </div>
        <div style="margin-top:12px;">
            <span class="score-badge" style="background:{color}20;color:{color};">
                {rationale.recommendation.value}
            </span>
        </div>
    </div>
    """


def create_loading_spinner() -> str:
    return """\
    <div style="display:flex;justify-content:center;padding:40px;">
        <div style="width:40px;height:40px;border:4px solid #e5e7eb;
             border-top-color:#3b82f6;border-radius:50%;
             animation:spin 0.8s linear infinite;"></div>
    </div>
    <style>
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
    """


def create_error_panel(message: str) -> str:
    """Return a prominent error message panel for display in the UI."""
    return f"""\
    <div style="border:2px solid #fecaca;background:#fef2f2;border-radius:10px;
                padding:20px;margin:12px 0;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <span style="font-size:24px;">&#9888;&#65039;</span>
            <strong style="color:#991b1b;font-size:16px;">Error</strong>
        </div>
        <p style="color:#b91c1c;margin:0;font-size:14px;line-height:1.5;">{message}</p>
    </div>"""
