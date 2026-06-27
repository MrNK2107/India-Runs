from __future__ import annotations

import math

# ruff: noqa: E501 — long HTML/CSS inline style strings are intentional
from src.core.models import MatchResult, MatchScores, Profile, Rationale, SearchResultItem


def _bullet_years(years: float | None) -> str:
    return f" \u2022 {years:.0f}yrs exp" if years else ""


MATCH_COLORS = {
    "strong_match": "#10b981",
    "good_match": "#3b82f6",
    "potential_match": "#f59e0b",
    "weak_match": "#ef4444",
}


_BAR_COLORS = {
    "Overall": "#b8a9c9",
    "Skill": "#8ab89e",
    "Experience": "#9a8ab0",
    "Semantic": "#a8ccb8",
    "Keyword": "#ccc09f",
    "Education": "#ccafb6",
    "AI Rerank": "#b5c8da",
    "Behavioral": "#b8929a",
    "Career": "#b8a87c",
    "Proficiency": "#b8a9c9",
}


def _bar_color(label: str) -> str:
    return _BAR_COLORS.get(label.strip(), "#b8a9c9")


def _score_bar(label: str, value: float, color: str | None = None) -> str:
    pct = max(0, min(100, int(value * 100)))
    c = color or _bar_color(label)
    return f"""
    <div class="score-bar-row">
        <span class="score-bar-label">{label}</span>
        <div class="score-bar-track">
            <div class="score-bar-fill" style="width:{pct}%;background:{c};"></div>
        </div>
        <span class="score-bar-pct">{pct}%</span>
    </div>"""


def _score_badge(value: float) -> str:
    pct = int(value * 100)
    if pct >= 70:
        css_class = "score-strong"
    elif pct >= 50:
        css_class = "score-good"
    elif pct >= 30:
        css_class = "score-potential"
    else:
        css_class = "score-weak"
    return f"""
    <div class="score-badge {css_class}">
        <div class="score-badge-value">{pct}</div>
        <div class="score-badge-label">{css_class.replace('score-', '')}</div>
    </div>"""


def create_candidate_card(item: SearchResultItem) -> str:
    s = item.scores if item.scores is not None else MatchScores()
    bars = ""
    bars += _score_bar("Overall", s.overall)
    bars += _score_bar("Skill", s.skill_match)
    bars += _score_bar("Experience", s.experience_match)
    bars += _score_bar("Semantic", s.semantic_similarity)
    bars += _score_bar("Keyword", s.keyword_match)
    if s.education_match is not None:
        bars += _score_bar("Education", s.education_match)
    if s.cross_encoder_score is not None:
        bars += _score_bar("AI Rerank", s.cross_encoder_score)
    if s.behavioral_score is not None:
        bars += _score_bar("Behavioral", s.behavioral_score)
    if s.career_trajectory_score is not None:
        bars += _score_bar("Career", s.career_trajectory_score)
    if s.skill_proficiency_score is not None:
        bars += _score_bar("Proficiency", s.skill_proficiency_score)

    skills_html = "".join(
        f'<span class="skill-chip matched">{skill}</span>' for skill in item.matched_skills[:8]
    )
    missing_html = "".join(
        f'<span class="skill-chip missing">{skill}</span>' for skill in item.missing_skills[:5]
    )

    return f"""
    <div class="candidate-card">
        <div class="candidate-header" style="display:flex;justify-content:space-between;align-items:start;gap:16px;">
            <div style="flex:1;">
                <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                    <span class="candidate-rank">#{item.rank}</span>
                    <span class="candidate-name">{item.name}</span>
                    <code style="font-size:10px;color:var(--text-muted);background:rgba(167,139,250,0.06);padding:1px 6px;border-radius:4px;font-family:'JetBrains Mono',monospace;">{item.profile_id}</code>
                </div>
                <div class="candidate-role">
                    {item.current_title or 'N/A'}
                    {(
                        f'<span style="opacity:0.45;font-weight:400;"> at </span>'
                        f'<strong>{item.current_company}</strong>'
                        if item.current_company else ""
                    )}
                </div>
                <div class="candidate-meta">
                    {item.location or 'Location N/A'}
                    {_bullet_years(item.experience_years)}
                </div>
            </div>
            {_score_badge(s.overall)}
        </div>
        <div style="margin-top:16px;">
            <div style="font-size:12px;font-weight:600;color:var(--text-secondary);margin-bottom:8px;letter-spacing:0.3px;">
                SCORE BREAKDOWN
                <span style="font-weight:400;color:var(--text-muted);font-size:11px;margin-left:6px;">confidence: {s.confidence:.0%}</span>
            </div>
            {bars}
        </div>
        <div style="margin-top:14px;display:flex;flex-wrap:wrap;gap:6px;">
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
        <polygon points="{outer_points}" fill="#f5f0e8" stroke="#e8e0d4" stroke-width="1"/>
        {grid_lines}
        <polygon points="{data_points}" fill="#d4c9e380" stroke="#b8a9c9" stroke-width="2"/>
        <circle cx="{cx}" cy="{cy}" r="2" fill="#b8a9c9"/>
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

    if not results_json or results_json.strip() in ("[]", "", "{}"):
        return create_empty_analytics()

    try:
        raw = json.loads(results_json) if results_json else []
    except (json.JSONDecodeError, TypeError):
        raw = []

    total = len(raw)
    listwise_ranked = any(r.get("listwise_ranked", False) for r in raw) if raw else False
    listwise_badge = '<span class="badge badge-listwise">🏆 Listwise Ranked</span>' if listwise_ranked else ""

    match_results = []
    for r in raw:
        if isinstance(r, dict):
            scores_dict = r.get("scores", {})
            if not isinstance(scores_dict, dict):
                scores_dict = {}
            match_scores = MatchScores(
                overall=float(scores_dict.get("overall", r.get("_re_score", 0)) or 0),
                semantic_similarity=float(scores_dict.get("semantic_similarity") or 0),
                keyword_match=float(scores_dict.get("keyword_match") or 0),
                skill_match=float(scores_dict.get("skill_match", 0) or 0),
                experience_match=float(scores_dict.get("experience_match", 0) or 0),
                location_match=float(scores_dict.get("location_match") or 0)
                if scores_dict.get("location_match") is not None else None,
                education_match=float(scores_dict.get("education_match") or 0)
                if scores_dict.get("education_match") is not None else None,
                confidence=float(scores_dict.get("confidence", 0) or 0),
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

    scores = [m.scores.overall for m in match_results if m.scores.overall > 0]
    if scores:
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        bins = [0] * 10
        for s in scores:
            idx = min(9, int(s * 10))
            bins[idx] += 1
        max_bin = max(bins) or 1
        pastel_colors = [
            "#fbcfe8", "#fed7aa", "#fde68a", "#a7f3d0",
            "#bfdbfe", "#c4b5fd", "#ddd6fe", "#fbcfe8",
            "#fed7aa", "#a7f3d0"
        ]
        bar_chart = "".join(
            f'<div style="flex:1;background:{pastel_colors[i]};'
            f'height:{max(4, int(b * 120 / max_bin))}px;'
            f'border-radius:4px 4px 0 0;transition:height 0.5s;" '
            f'title="{int(i*10)}-{int(i*10+9)}%: {b} candidates"></div>'
            for i, b in enumerate(bins)
        )
    else:
        avg_score = max_score = min_score = 0
        bar_chart = '<div style="text-align:center;padding:30px;color:var(--text-muted);">No scores to display</div>'

    metric_cards = ""
    if len(match_results) >= 3:
        profiledict = _get_bias_profiles(match_results)
        fairness = compute_all_fairness_metrics(match_results, profiledict)
        dp = fairness.get("demographic_parity", {})
        lang_bias = fairness.get("language_bias", {})

        def _metric_card(label, value, threshold, format_str="{:.3f}"):
            val = value if isinstance(value, (int, float)) else 0
            if val < threshold:
                cls = "pastel-green"
                color = "#059669"
                status = "✅ No bias detected"
            elif val < threshold * 2:
                cls = "pastel-amber"
                color = "#d97706"
                status = "👀 Monitor closely"
            else:
                cls = "pastel-rose"
                color = "#e11d48"
                status = "⚠️ Bias detected"
            return f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color};">{format_str.format(val)}</div>
                <div style="font-size:12px;color:var(--text-muted);margin-top:4px;">{status}</div>
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

    return f"""
    <div style="padding:8px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
            <h3 style="margin:0;font-size:20px;font-weight:700;color:var(--text-primary);">Fairness &amp; Bias Metrics</h3>
            <div style="display:flex;gap:8px;align-items:center;">
                {listwise_badge}
                <span class="badge badge-pii">🔒 PII Anonymized</span>
                <span style="font-size:12px;color:var(--text-muted);">{total} candidates</span>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;">
            {metric_cards or '<div class="metric-card"><div class="metric-label">Run a search to see metrics</div></div>'}
        </div>

        <h3 style="margin:28px 0 16px;font-size:18px;font-weight:700;color:var(--text-primary);">Score Distribution</h3>
        <div style="display:flex;align-items:flex-end;gap:4px;height:130px;padding:0 4px;">
            {bar_chart}
        </div>
        <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-muted);margin-top:4px;">
            <span>0%</span><span>50%</span><span>100%</span>
        </div>
        <div style="display:flex;gap:24px;margin-top:12px;font-size:13px;color:var(--text-secondary);">
            <span>Avg: <strong style="color:var(--text-primary);">{avg_score:.1%}</strong></span>
            <span>Max: <strong style="color:var(--text-primary);">{max_score:.1%}</strong></span>
            <span>Min: <strong style="color:var(--text-primary);">{min_score:.1%}</strong></span>
        </div>

        {_build_distribution_table(match_results)}
    </div>
    """


def _build_distribution_table(match_results: list[MatchResult]) -> str:
    strong = sum(1 for m in match_results if m.scores.overall >= 0.8)
    good = sum(1 for m in match_results if 0.6 <= m.scores.overall < 0.8)
    potential = sum(1 for m in match_results if 0.4 <= m.scores.overall < 0.6)
    weak = sum(1 for m in match_results if m.scores.overall < 0.4)
    total = len(match_results) or 1
    return f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px;">
        <div style="background:rgba(52,211,153,0.1);padding:16px;border-radius:var(--radius-md);text-align:center;border:1px solid rgba(52,211,153,0.15);">
            <div style="font-size:28px;font-weight:800;color:#059669;">{strong}</div>
            <div style="font-size:12px;color:var(--text-muted);margin-top:2px;">Strong ({strong*100//total}%)</div>
        </div>
        <div style="background:rgba(96,165,250,0.1);padding:16px;border-radius:var(--radius-md);text-align:center;border:1px solid rgba(96,165,250,0.15);">
            <div style="font-size:28px;font-weight:800;color:#2563eb;">{good}</div>
            <div style="font-size:12px;color:var(--text-muted);margin-top:2px;">Good ({good*100//total}%)</div>
        </div>
        <div style="background:rgba(251,191,36,0.1);padding:16px;border-radius:var(--radius-md);text-align:center;border:1px solid rgba(251,191,36,0.15);">
            <div style="font-size:28px;font-weight:800;color:#d97706;">{potential}</div>
            <div style="font-size:12px;color:var(--text-muted);margin-top:2px;">Potential ({potential*100//total}%)</div>
        </div>
        <div style="background:rgba(251,113,133,0.1);padding:16px;border-radius:var(--radius-md);text-align:center;border:1px solid rgba(251,113,133,0.15);">
            <div style="font-size:28px;font-weight:800;color:#e11d48;">{weak}</div>
            <div style="font-size:12px;color:var(--text-muted);margin-top:2px;">Weak ({weak*100//total}%)</div>
        </div>
    </div>"""


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

    color = MATCH_COLORS.get(rationale.recommendation.value, "#a78bfa")
    summary = rationale.summary or "No summary available."

    strengths_html = "".join(
        f"<li>{s}</li>" for s in rationale.strengths[:5]
    )
    gaps_html = "".join(
        f"<li>{g}</li>" for g in rationale.gaps[:5]
    )

    return f"""
    <div class="candidate-card rationale-panel" style="border-left-color:{color};">
        <h4 style="margin:0 0 4px;font-size:15px;font-weight:600;color:var(--text-primary);">Rationale: {profile_summary}</h4>
        <p style="color:var(--text-secondary);font-size:13px;line-height:1.6;margin:6px 0;">{summary}</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px;">
            <div>
                <strong style="color:#059669;font-size:13px;">✓ Strengths</strong>
                <ul style="margin:6px 0;padding-left:18px;font-size:12px;color:var(--text-secondary);line-height:1.6;">{strengths_html}</ul>
            </div>
            <div>
                <strong style="color:#e11d48;font-size:13px;">✗ Gaps</strong>
                <ul style="margin:6px 0;padding-left:18px;font-size:12px;color:var(--text-secondary);line-height:1.6;">{gaps_html}</ul>
            </div>
        </div>
        <div style="margin-top:8px;">
            <span class="badge badge-listwise" style="background:{color}15;color:{color};border-color:{color}30;">
                {rationale.recommendation.value}
            </span>
        </div>
    </div>
    """


# ── Progressive Loading Steps ──────────────────────────────────────────
LOADING_STEPS = [
    ("🔍", "Parsing query", "Understanding job requirements, skills, and context"),
    ("📡", "Searching index", "Scanning 100K+ profiles with hybrid search"),
    ("⚡", "AI reranking", "Cross-encoder scoring for precision matching"),
    ("📊", "Computing scores", "Multi-signal evaluation across 6 dimensions"),
    ("🎯", "Building results", "Assembling ranked shortlist with rationales"),
]

LOADING_STEP_TIMING = [0.15, 0.40, 0.60, 0.80, 1.0]


def _step_class(i: int, current_step: int) -> str:
    if current_step < 0:
        return "completed"
    if i < current_step:
        return "completed"
    if i == current_step:
        return "active"
    return ""


def _step_extra(i: int, current_step: int, desc: str) -> str:
    if i == current_step and 0 <= current_step < len(LOADING_STEPS):
        return f'<br><span style="font-size:10px;color:var(--text-muted);">{desc}</span>'
    return ""


def create_progress_html(current_step: int = 0) -> str:
    if current_step < 0:
        steps_html = "".join(
            f'<div class="progress-step completed">'
            f'<span class="step-icon">{icon}</span>{label}</div>'
            for icon, label, _ in LOADING_STEPS
        )
        pct = 100
    else:
        steps_html = "".join(
            f'<div class="progress-step {_step_class(i, current_step)}">'
            f'<span class="step-icon">{icon}</span>{label}'
            f'{_step_extra(i, current_step, desc)}'
            f'</div>'
            for i, (icon, label, desc) in enumerate(LOADING_STEPS)
        )
        pct = int(sum(LOADING_STEP_TIMING[:current_step + 1]) / len(LOADING_STEPS) * 100)

    step_label = LOADING_STEPS[current_step][1] if 0 <= current_step < len(LOADING_STEPS) else "Complete"
    step_desc = LOADING_STEPS[current_step][2] if 0 <= current_step < len(LOADING_STEPS) else ""

    icon = "✅" if current_step < 0 else "⏳"
    return f"""
    <div class="progress-container">
        <div class="progress-header">
            <div class="progress-title">
                <span>{icon}</span>
                {step_label}
            </div>
            <span class="progress-step-label">{step_desc}</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill" style="width:{pct}%;"></div>
        </div>
        <div class="progress-steps">
            {steps_html}
        </div>
    </div>
    """


def create_loading_overlay(message: str = "Searching candidates...") -> str:
    return f"""\
    <div class="loading-overlay">
        <div class="loading-ring-container">
            <div class="loading-ring"></div>
            <div class="loading-ring"></div>
            <div class="loading-ring"></div>
        </div>
        <div class="loading-text">{message}</div>
        <div class="loading-sub">This may take 10-30 seconds for deep search</div>
        <div class="loading-particles">
            <div class="loading-particle"></div>
            <div class="loading-particle"></div>
            <div class="loading-particle"></div>
            <div class="loading-particle"></div>
            <div class="loading-particle"></div>
        </div>
    </div>
    """


def create_empty_state() -> str:
    return """\
    <div class="empty-state">
        <div class="empty-state-icon" style="animation:float 3s ease-in-out infinite;">🔍</div>
        <div class="empty-state-title">Ready to find talent</div>
        <div class="empty-state-desc">
            Describe the ideal candidate on the left and click <strong>Search Candidates</strong>
            to find matching profiles.
        </div>
        <div class="empty-state-hint">
            Adjust scoring sliders in the sidebar to fine-tune results
        </div>
    </div>
    """


def create_error_panel(message: str) -> str:
    """Return a prominent error message panel for display in the UI."""
    return f"""\
    <div style="border:1.5px solid var(--red-100);background:rgba(254,226,226,0.15);backdrop-filter:blur(8px);border-radius:var(--radius-md);padding:20px;margin:12px 0;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
            <span style="font-size:22px;">⚠️</span>
            <strong style="color:var(--red-700);font-size:15px;">Error</strong>
        </div>
        <p style="color:var(--red-700);margin:0;font-size:13px;line-height:1.6;">{message}</p>
    </div>"""


def create_empty_analytics() -> str:
    return """\
    <div class="empty-state">
        <div class="empty-state-icon" style="animation:float 3s ease-in-out infinite;">📊</div>
        <div class="empty-state-title">No results yet</div>
        <div class="empty-state-desc">Run a search first to see analytics and fairness metrics.</div>
    </div>
    """
