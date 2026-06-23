from __future__ import annotations

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

from src.api.routes.search import init_orchestrator
from src.core.config import DATA_DIR
from src.core.models import MatchScores, SearchResultItem
from src.core.profile_store import ProfileStore
from src.matching.scorer import DEFAULT_SLIDER_WEIGHTS, CandidateScorer
from src.ui.components import (
    create_analytics_dashboard,
    create_candidate_card,
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

    from src.agents.executor import ExecutorAgent
    from src.agents.orchestrator import Orchestrator
    from src.agents.planner import PlannerAgent
    from src.agents.reflector import ReflectorAgent
    from src.language.multilingual import MultilingualEmbedder
    from src.search.bm25_search import BM25Search
    from src.search.hybrid import HybridSearch
    from src.search.reranker import CrossEncoderReranker
    from src.search.vector_search import VectorSearch

    embedder = MultilingualEmbedder()
    _ = embedder.model

    vector_search = VectorSearch()
    vector_search.load(faiss_path, id_map_path)

    bm25_search = BM25Search()
    bm25_search.load(bm25_path)

    hybrid_search = HybridSearch(vector_search, bm25_search, embedder)
    reranker = CrossEncoderReranker()
    _ = reranker.model  # Lazy load — disabled by default, ok if None
    scorer = CandidateScorer()

    profiles = ProfileStore()
    offset_index_path = DATA_DIR / "indexes" / "offset_index.json"
    if offset_index_path.exists():
        profiles.load_offset_index(offset_index_path)
    sample_path = DATA_DIR / "samples" / "sample_candidates.json"
    if sample_path.exists():
        profiles.load_sample(sample_path)

    planner = PlannerAgent()
    executor = ExecutorAgent(hybrid_search, reranker, scorer, profiles)
    reflector = ReflectorAgent()
    orchestrator = Orchestrator(planner, executor, reflector)
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
    *slider_values: float,
) -> tuple[str, str, str]:
    if not query.strip():
        return (
            "<div style='padding:40px;text-align:center;color:#9ca3af;'>"
            "<p style='font-size:18px;margin-bottom:8px;'>&#128269; Enter a query to search</p>"
            "<p style='font-size:14px;'>Describe the ideal candidate — skills, experience, location.</p>"
            "</div>",
            "",
            "[]",
        )

    from src.ui.components import create_error_panel

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

    try:
        t0 = time.time()
        response = await _orchestrator.run(query, slider_weights=slider_weights)
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
            )
            html += create_candidate_card(item)
        html += "</div>"

        return html
    except Exception as e:
        logger.exception("Re-rank processing failed")
        return create_error_panel(f"Re-ranking failed: {e}")


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="India Runs — Intelligent Candidate Discovery",
    ) as app:
        gr.Markdown("# India Runs — Intelligent Candidate Discovery")
        gr.Markdown("*Beyond keywords. Beyond filters. AI that understands hiring.*")

        results_state = gr.State("")

        with gr.Tabs():
            with gr.Tab("Search"):
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Job Query",
                            placeholder="e.g., senior DevOps engineer with 5+ yrs AWS...",
                            lines=3,
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

                        search_btn = gr.Button("Search Candidates", variant="primary", size="lg")

                        with gr.Accordion("Scoring Weights", open=True):
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

                    with gr.Column(scale=1):
                        location_filter = gr.Textbox(label="Location")
                        experience_filter = gr.Slider(
                            label="Min Experience (years)", minimum=0, maximum=20, step=1, value=0,
                        )
                        remote_ok = gr.Checkbox(label="Remote OK", value=False)
                        max_results = gr.Slider(
                            label="Max Results", minimum=5, maximum=50, step=5, value=10,
                        )

                results_area = gr.HTML(
                    label="Results",
                    value="<div style='padding:40px;text-align:center;color:#9ca3af;'>"
                          "<p style='font-size:18px;margin-bottom:8px;'>&#128269; No search yet</p>"
                          "<p style='font-size:14px;'>Enter a query above and click "
                          "<strong>Search Candidates</strong> to find matching profiles.</p>"
                          "<p style='font-size:12px;color:#d1d5db;margin-top:16px;'>"
                          "Adjust the scoring sliders to fine-tune results.</p>"
                          "</div>",
                )
                rationale_area = gr.HTML(label="Rationale Report", value="")

                search_inputs = [
                    query_input, location_filter,
                    experience_filter, remote_ok, max_results,
                    *slider_inputs,
                ]
                search_btn.click(
                    fn=search_handler,
                    inputs=search_inputs,
                    outputs=[results_area, rationale_area, results_state],
                )

                re_rank_btn = gr.Button("Re-Rank with Current Weights", variant="secondary")
                re_rank_inputs = [results_state, *slider_inputs]
                re_rank_btn.click(
                    fn=re_rank_handler,
                    inputs=re_rank_inputs,
                    outputs=[results_area],
                )
                for slider in slider_inputs:
                    slider.change(
                        fn=re_rank_handler,
                        inputs=re_rank_inputs,
                        outputs=[results_area],
                    )

            with gr.Tab("Analytics"):
                analytics_html = gr.HTML(label="Analytics Dashboard")
                refresh_btn = gr.Button("Refresh Analytics", variant="secondary")
                refresh_btn.click(
                    fn=create_analytics_dashboard,
                    inputs=[results_state],
                    outputs=[analytics_html],
                )
                search_btn.click(
                    fn=create_analytics_dashboard,
                    inputs=[results_state],
                    outputs=[analytics_html],
                )

            with gr.Tab("About"):
                about_lines = [
                    "## About This System",
                    "",
                    "**Intelligent Candidate Discovery** - a hybrid semantic search system",
                    "that goes beyond keyword matching.",
                    "",
                    "### Architecture",
                    "- **Hybrid Search**: BM25 + FAISS vector search + Reciprocal Rank Fusion",
                    "- **Cross-Encoder Reranking**: MiniLM for precision",
                    "- **Agentic Workflow**: Plan -> Execute -> Reflect -> Re-plan (LangGraph)",
                    "- **Multilingual**: 30+ Indian languages via multilingual embeddings",
                    "- **Rationale Reports**: Every match comes with an explanation",
                    "",
                    "### Interactive Scoring",
                    "- Adjust **6 recruiter-facing dimensions** via sliders",
                    "- Results re-rank **instantly** without re-searching",
                    "- Fine-tune for each role's unique priorities",
                    "",
                    "### Fairness",
                    "- Bias monitoring across demographics, location, university",
                    "- PII anonymization to prevent name-based bias",
                    "- Transparent, explainable rankings",
                    "",
                    "### Tech Stack",
                    "- FastAPI, FAISS, sentence-transformers, LangGraph, Gradio",
                    "",
                    "### Hackathon",
                    "- Built for India Runs - Track 1: Data & AI Challenge",
                ]
                gr.Markdown("\n".join(about_lines))

    return app


app = create_app()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css="src/ui/styles.css",
    )
