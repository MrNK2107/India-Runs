from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import gradio as gr

# Ensure project root is on the path for direct execution
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.config import DATA_DIR  # noqa: E402
from src.core.models import MatchScores, Rationale, SearchResultItem  # noqa: E402
from src.matching.scorer import DEFAULT_SLIDER_WEIGHTS, CandidateScorer  # noqa: E402
from src.ui.components import (  # noqa: E402
    LOADING_STEPS,
    create_analytics_dashboard,
    create_candidate_card,
    create_empty_state,
    create_loading_overlay,
    create_progress_html,
    create_rationale_panel,
)

logger = logging.getLogger(__name__)

indexes_dir = DATA_DIR / "indexes"
faiss_path = indexes_dir / "faiss_index.bin"
id_map_path = indexes_dir / "faiss_id_map.json"
bm25_path = indexes_dir / "bm25_index.pkl"

_search_initialized = False


def _ensure_search_system() -> bool:
    """Lazy-initialize heavy model components (embeddings, FAISS, cross-encoder)
    on first search instead of loading at import time."""
    global _search_initialized
    if _search_initialized:
        return True
    if not faiss_path.exists():
        logger.warning("No FAISS index found. Run 'python scripts/build_indexes.py' first.")
        return False

    from src.api.routes.search import init_orchestrator
    from src.core.config import build_orchestrator

    orchestrator, _, _ = build_orchestrator(
        faiss_path=faiss_path,
        id_map_path=id_map_path,
        bm25_path=bm25_path,
    )
    init_orchestrator(orchestrator)
    logger.info("Search system initialized")
    _search_initialized = True
    return True

SLIDER_DIMS = [
    ("Skill Match", "skill_match", DEFAULT_SLIDER_WEIGHTS["skill_match"]),
    ("Experience", "experience_match", DEFAULT_SLIDER_WEIGHTS["experience_match"]),
    ("Education", "education_match", DEFAULT_SLIDER_WEIGHTS["education_match"]),
    ("Assessment", "assessment_score", DEFAULT_SLIDER_WEIGHTS["assessment_score"]),
    ("Behavioral", "behavioral_signals", DEFAULT_SLIDER_WEIGHTS["behavioral_signals"]),
    ("Cultural Fit", "cultural_fit", DEFAULT_SLIDER_WEIGHTS["cultural_fit"]),
]

SLIDER_KEYS = [k for _, k, _ in SLIDER_DIMS]


def _parse_slider_weights(*slider_values: float) -> dict[str, float]:
    return {key: val for key, val in zip(SLIDER_KEYS, slider_values)}


async def search_handler(
    query: str, location: str, min_experience: int, remote_ok: bool, max_results: int,
    use_turbo: bool,
    *slider_values: float,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str]:
    if not query.strip():
        return (
            create_empty_state(),
            "",
            "[]",
        )

    from src.ui.components import create_error_panel

    progress(0.05, desc="🔍 Initializing search system...")
    if not _ensure_search_system():
        return (
            create_error_panel(
                "Search system not initialized. Please build indexes first "
                "by running <tt>python scripts/build_indexes.py</tt>."
            ),
            "",
            "[]",
        )

    slider_weights = _parse_slider_weights(*slider_values)

    from src.api.routes.search import _orchestrator
    from src.core.models import SearchFilters

    filters = SearchFilters(
        location=location.strip() if location.strip() else None,
        min_experience_years=float(min_experience) if min_experience > 0 else None,
        remote_ok=bool(remote_ok),
    )

    try:
        t0 = time.time()

        progress(0.15, desc="📝 Parsing query — understanding skills, experience, location...")
        await asyncio.sleep(0.01)  # Let progress render

        progress(0.30, desc="📡 Hybrid search — scanning 100K profiles (FAISS + BM25)...")
        await asyncio.sleep(0.01)

        response = await _orchestrator.run(
            query,
            slider_weights=slider_weights,
            use_turbo=use_turbo,
            top_k=max_results,
            filters=filters,
        )

        progress(0.65, desc="⚡ AI reranking — cross-encoder precision scoring...")
        await asyncio.sleep(0.01)

        progress(0.80, desc="📊 Computing multi-signal scores across 6 dimensions...")
        await asyncio.sleep(0.01)

        elapsed = time.time() - t0
        logger.info(f"Search completed in {elapsed:.1f}s")
    except Exception as e:
        logger.exception("Search failed")
        return (
            create_error_panel(f"Search failed: {e}"),
            "",
            "[]",
        )

    # Serialize results to JSON for caching in Gradio State
    raw_results = []
    for item in response.results[:max_results]:
        raw_results.append({
            "rank": item.rank,
            "profile_id": item.profile_id,
            "name": item.name,
            "current_title": item.current_title,
            "current_company": item.current_company,
            "location": item.location,
            "experience_years": item.experience_years,
            "scores": item.scores.model_dump() if hasattr(item.scores, "model_dump") else {},
            "matched_skills": item.matched_skills,
            "missing_skills": item.missing_skills,
            "rationale": (
                item.rationale.model_dump() if hasattr(item.rationale, "model_dump") else {}
            ),
        })

    results_json = json.dumps(raw_results)

    results_html = "<div class='results-container'>"
    for item in response.results[:max_results]:
        results_html += create_candidate_card(item)
    results_html += "</div>"

    # Metadata header
    md = response.search_metadata
    methods_str = " + ".join(md.methods_used) if md and md.methods_used else "hybrid"
    badges = ""
    if md and md.listwise_ranked:
        badges += '<span class="badge badge-listwise">&#127942; Listwise Ranked</span>'
    if md and md.pii_anonymized:
        badges += '<span class="badge badge-pii">&#128737;&#65039; PII Anonymized</span>'
    badges += f'<span class="badge badge-method">{methods_str}</span>'
    metadata_header = f"""
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:12px;flex-wrap:wrap;">
        {badges}
        <span style="font-size:12px;color:#9ca3af;margin-left:auto;">
            {response.total_candidates_searched} candidates searched
            | {response.processing_time_ms}ms
        </span>
    </div>
    """

    rationales_html = ""
    for item in response.results[:5]:
        rationales_html += create_rationale_panel(item.rationale, item.name)

    return metadata_header + results_html, rationales_html, results_json


def re_rank_handler(results_json: str, *slider_values: float) -> str:
    if not results_json or results_json == "[]":
        return "<p>No results to re-rank. Search first.</p>"

    from src.ui.components import create_error_panel

    try:
        raw = json.loads(results_json)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Re-rank received invalid JSON: %s", e)
        return create_error_panel("Could not parse cached results. Please re-run your search.")

    try:
        slider_weights = _parse_slider_weights(*slider_values)
    except Exception as e:
        logger.warning("Slider parsing error in re_rank_handler: %s", e)
        return create_error_panel(f"Could not parse slider weights: {e}")

    scorer = CandidateScorer()

    try:
        for r in raw:
            scores_dict = r.get("scores", {})
            match_scores = scorer.compute_overall(scores_dict, slider_weights)
            r["_re_score"] = match_scores.overall
            r["_re_scores"] = match_scores

        raw.sort(key=lambda x: x.get("_re_score", 0), reverse=True)

        html = "<div class='results-container'>"
        for rank, r in enumerate(raw, 1):
            r["rank"] = rank
            rationale_dict = r.get("rationale", {}) or {}
            item = SearchResultItem(
                rank=rank,
                profile_id=r.get("profile_id", ""),
                name=r.get("name", ""),
                current_title=r.get("current_title"),
                current_company=r.get("current_company"),
                location=r.get("location"),
                experience_years=r.get("experience_years"),
                scores=r.get("_re_scores", MatchScores()),
                matched_skills=r.get("matched_skills", []),
                missing_skills=r.get("missing_skills", []),
                rationale=Rationale(**rationale_dict) if isinstance(rationale_dict, dict) else Rationale(),
            )
            html += create_candidate_card(item)
        html += "</div>"

        return html
    except Exception as e:
        logger.exception("Re-rank processing failed")
        return create_error_panel(f"Re-ranking failed: {e}")


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="India Runs — AI-Powered Candidate Discovery",
    ) as app:
        gr.HTML("""
        <div class="app-header">
            <div class="app-title">India Runs</div>
            <div class="app-subtitle">AI-Powered Candidate Discovery — Beyond keywords, beyond filters.</div>
        </div>
        """)

        results_state = gr.State("")

        with gr.Tabs():
            with gr.Tab("🔍 Search"):
                with gr.Row():
                    with gr.Column(scale=1):
                        query_input = gr.Textbox(
                            label="Job Query",
                            placeholder="e.g., senior DevOps engineer with 5+ yrs AWS...",
                            lines=3,
                        )
                        use_turbo_toggle = gr.Checkbox(
                            label="⚡ Turbo Mode (Skip LLM planner/agent loops)",
                            value=False,
                        )
                        gr.Examples(
                            examples=[
                                "Find a senior Python developer with ML experience in Bangalore",
                                "aws devops engineer kubernetes terraform ci/cd",
                                "Product manager with B2B SaaS experience and growth mindset",
                                "Senior frontend engineer react typescript remote",
                            ],
                            inputs=query_input,
                        )
                        location_filter = gr.Textbox(label="📍 Location")
                        experience_filter = gr.Slider(
                            label="📅 Min Experience (years)", minimum=0, maximum=20, step=1, value=0,
                        )
                        remote_ok = gr.Checkbox(label="🏠 Remote OK", value=False)
                        max_results = gr.Slider(
                            label="📋 Max Results", minimum=5, maximum=50, step=5, value=10,
                        )

                        with gr.Accordion("🎛️ Scoring Weights", open=False):
                            gr.Markdown(
                                "Adjust the importance of each dimension. "
                                "Results re-rank automatically after search."
                            )
                            slider_inputs = []
                            for label, key, default in SLIDER_DIMS:
                                slider = gr.Slider(
                                    minimum=0, maximum=100, step=5, value=int(default * 100),
                                    label=label,
                                )
                                slider_inputs.append(slider)

                        search_btn = gr.Button("🔎 Search Candidates", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        results_area = gr.HTML(
                            label="Results",
                            value=create_empty_state(),
                        )
                        rationale_area = gr.HTML(label="Rationale Report", value="")
                        re_rank_btn = gr.Button("🔄 Re-Rank with Current Weights", variant="secondary")

                search_inputs = [
                    query_input, location_filter,
                    experience_filter, remote_ok, max_results,
                    use_turbo_toggle,
                    *slider_inputs,
                ]
                search_btn.click(
                    fn=lambda: create_loading_overlay("Initializing search..."),
                    outputs=[results_area],
                    show_progress="hidden",
                ).then(
                    fn=search_handler,
                    inputs=search_inputs,
                    outputs=[results_area, rationale_area, results_state],
                    show_progress="full",
                )

                re_rank_inputs = [results_state, *slider_inputs]
                re_rank_btn.click(
                    fn=re_rank_handler,
                    inputs=re_rank_inputs,
                    outputs=[results_area],
                    show_progress="hidden",
                )
                for slider in slider_inputs:
                    slider.change(
                        fn=re_rank_handler,
                        inputs=re_rank_inputs,
                        outputs=[results_area],
                        show_progress="hidden",
                    )

            with gr.Tab("📊 Analytics"):
                analytics_html = gr.HTML(label="Analytics Dashboard")
                refresh_btn = gr.Button("Refresh Analytics", variant="secondary")
                search_btn.click(
                    fn=create_analytics_dashboard,
                    inputs=[results_state],
                    outputs=[analytics_html],
                    show_progress="hidden",
                )
                refresh_btn.click(
                    fn=create_analytics_dashboard,
                    inputs=[results_state],
                    outputs=[analytics_html],
                    show_progress="hidden",
                )

            with gr.Tab("ℹ️ About"):
                gr.HTML("""
                <div class="about-section">
                    <h2>About This System</h2>
                    <p style="font-size:15px;"><strong>Intelligent Candidate Discovery</strong> — a hybrid semantic search system that goes beyond keyword matching.</p>

                    <h3>Architecture</h3>
                    <div class="about-feature-grid">
                        <div class="about-feature-card">
                            <strong>🔍 Hybrid Search</strong>
                            <span>BM25 + FAISS vector search + Reciprocal Rank Fusion for maximum recall</span>
                        </div>
                        <div class="about-feature-card">
                            <strong>⚡ Cross-Encoder</strong>
                            <span>MiniLM-L6 reranker for precision re-ranking of top candidates</span>
                        </div>
                        <div class="about-feature-card">
                            <strong>🧠 Agentic Workflow</strong>
                            <span>LangGraph: Plan → Execute → Reflect → Re-plan with LLM reasoning</span>
                        </div>
                        <div class="about-feature-card">
                            <strong>🌐 Multilingual</strong>
                            <span>30+ Indian languages via paraphrase-multilingual-MiniLM embeddings</span>
                        </div>
                        <div class="about-feature-card">
                            <strong>📋 Listwise Ranking</strong>
                            <span>Plackett-Luce tournament ranking for nuanced candidate comparison</span>
                        </div>
                        <div class="about-feature-card">
                            <strong>📝 Rationale Reports</strong>
                            <span>Every match includes human-readable explanations and evidence</span>
                        </div>
                    </div>

                    <h3>Interactive Scoring</h3>
                    <p>Adjust 6 recruiter-facing dimensions via sliders. Results re-rank instantly without re-searching. Fine-tune for each role's unique priorities.</p>

                    <h3>Fairness First</h3>
                    <p>Bias monitoring across demographics, location, and university. PII anonymization prevents name-based bias. Transparent, explainable rankings with fairness metrics in every search.</p>

                    <h3>Tech Stack</h3>
                    <p>FastAPI · FAISS · Sentence-Transformers · LangGraph · Gradio · Python</p>

                    <h3>🏆 India Runs — Track 1: Data & AI Challenge</h3>
                    <p>Built by Team Atlas — Nikhil Choudhary</p>
                </div>
                """)

    return app


app = create_app()

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css="src/ui/styles.css",
    )
