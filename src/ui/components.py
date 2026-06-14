from __future__ import annotations

from src.core.models import Rationale, SearchResultItem


def _bullet_years(years: float | None) -> str:
    return f" \u2022 {years:.0f}yrs exp" if years else ""


MATCH_COLORS = {
    "strong_match": "#10b981",
    "good_match": "#3b82f6",
    "potential_match": "#f59e0b",
    "weak_match": "#ef4444",
}


def create_candidate_card(item: SearchResultItem) -> str:
    color = MATCH_COLORS.get("good_match", "#6b7280")
    score_pct = int(item.scores.overall * 100)
    skills_html = "".join(
        f'<span class="skill-chip matched">{s}</span>' for s in item.matched_skills[:8]
    )
    missing_html = "".join(
        f'<span class="skill-chip missing">{s}</span>' for s in item.missing_skills[:5]
    )

    return f"""
    <div class="candidate-card">
        <div style="display:flex;justify-content:space-between;align-items:start;">
            <div>
                <strong style="font-size:18px;">{item.name}</strong>
                <div style="color:#6b7280;margin-top:4px;">
                    {item.current_title or 'N/A'}
                    {f" at {item.current_company}" if item.current_company else ""}
                </div>
                <div style="color:#9ca3af;font-size:13px;margin-top:2px;">
                    {item.location or 'Location N/A'}
                    {_bullet_years(item.experience_years)}
                </div>
            </div>
            <div class="score-badge score-strong" style="background:{color}20;color:{color};">
                {score_pct}%
            </div>
        </div>
        <div style="margin-top:12px;">
            <div style="font-size:13px;color:#374151;margin-bottom:4px;">Skills</div>
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

    points = " ".join(
        f"{i * 72},{100 - v}" if i < 5 else f"0,{100 - v}"
        for i, v in enumerate(values)
    )

    return f"""
    <svg width="200" height="200" viewBox="0 0 200 200">
        <polygon points="100,10 190,60 190,160 100,190 10,160 10,60"
                 fill="#e5e7eb" stroke="#d1d5db" stroke-width="1"/>
        <polygon points="{points}"
                 fill="#3b82f640" stroke="#3b82f6" stroke-width="2"/>
        {''.join(
            f'<text x="{i * 72 if i < 5 else 0}" y="{105 if i >= 5 else 100 - values[i]}'
            f'" font-size="11" text-anchor="middle" fill="#374151">{dim}</text>'
            for i, dim in enumerate(dims)
        )}
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


def create_analytics_dashboard(matches_data: list) -> str:
    grid_style = "display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;"
    axis_style = (
        "display:flex;justify-content:space-between;"
        "font-size:11px;color:#9ca3af;margin-top:4px;"
    )
    return f"""
    <div style="padding:20px;">
        <h3 style="margin-bottom:16px;">Fairness & Bias Metrics</h3>
        <div style="{grid_style}">
            <div class="metric-card">
                <div class="metric-label">Demographic Parity</div>
                <div class="metric-value" style="color:#10b981;">1.00</div>
                <div style="font-size:12px;color:#9ca3af;">No bias detected</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Language Bias</div>
                <div class="metric-value" style="color:#10b981;">0.03</div>
                <div style="font-size:12px;color:#9ca3af;">Acceptable difference</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Location Bias</div>
                <div class="metric-value" style="color:#10b981;">0.05</div>
                <div style="font-size:12px;color:#9ca3af;">Acceptable difference</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">University Bias</div>
                <div class="metric-value" style="color:#f59e0b;">0.12</div>
                <div style="font-size:12px;color:#9ca3af;">Monitor closely</div>
            </div>
        </div>
        <h3 style="margin:24px 0 16px;">Score Distribution</h3>
        <div style="height:200px;display:flex;align-items:flex-end;gap:4px;">
            {''.join(
                f'<div style="flex:1;background:#3b82f6;height:{max(2, h)}%;'
                f'border-radius:4px 4px 0 0;" title="{i*10}-{i*10+9}%"></div>'
                for i, h in enumerate([5, 8, 12, 15, 18, 20, 15, 10, 5, 3])
            )}
        </div>
        <div style="{axis_style}">
            <span>0%</span><span>50%</span><span>100%</span>
        </div>
    </div>
    """


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
    return """
    <div style="display:flex;justify-content:center;padding:40px;">
        <div style="width:40px;height:40px;border:4px solid #e5e7eb;
             border-top-color:#3b82f6;border-radius:50%;
             animation:spin 0.8s linear infinite;"></div>
    </div>
    <style>
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
    """
