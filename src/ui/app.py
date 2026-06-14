from __future__ import annotations

import logging

import gradio as gr

from src.ui.components import (
    create_analytics_dashboard,
    create_candidate_card,
    create_rationale_panel,
)

logger = logging.getLogger(__name__)


async def search_handler(
    query: str, location: str, min_experience: int, remote_ok: bool, max_results: int,
) -> tuple[str, str]:
    if not query.strip():
        return "<p style='color: #ef4444;'>Please enter a search query.</p>", ""

    from src.api.routes.search import _orchestrator

    if _orchestrator is None:
        return (
            "<p style='color: #f59e0b;'>Search system not initialized. "
            "Please build indexes first.</p>",
            "",
        )

    try:
        response = await _orchestrator.run(query)
    except Exception as e:
        logger.exception("Search failed")
        return f"<p style='color: #ef4444;'>Search failed: {e}</p>", ""

    results_html = "<div class='results-container'>"
    for item in response.results[:max_results]:
        results_html += create_candidate_card(item)
    results_html += "</div>"

    rationales_html = ""
    for item in response.results[:5]:
        rationales_html += create_rationale_panel(item.rationale, item.name)

    return results_html, rationales_html


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="India Runs \u2014 Intelligent Candidate Discovery",
        theme=gr.themes.Soft(),
        css="src/ui/styles.css",
    ) as app:
        gr.Markdown("# India Runs \u2014 Intelligent Candidate Discovery")
        gr.Markdown("*Beyond keywords. Beyond filters. AI that understands hiring.*")

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
                    with gr.Column(scale=1):
                        location_filter = gr.Textbox(label="Location")
                        experience_filter = gr.Slider(
                            label="Min Experience (years)", minimum=0, maximum=20, step=1, value=0,
                        )
                        remote_ok = gr.Checkbox(label="Remote OK", value=False)
                        max_results = gr.Slider(
                            label="Max Results", minimum=5, maximum=50, step=5, value=10,
                        )

                search_btn = gr.Button("Search Candidates", variant="primary", size="lg")
                results_area = gr.HTML(label="Results")
                rationale_area = gr.HTML(label="Rationale Report")

                search_btn.click(
                    fn=search_handler,
                    inputs=[
                        query_input, location_filter,
                        experience_filter, remote_ok, max_results,
                    ],
                    outputs=[results_area, rationale_area],
                )

            with gr.Tab("Analytics"):
                analytics_html = gr.HTML(label="Analytics Dashboard")
                refresh_btn = gr.Button("Refresh Analytics", variant="secondary")
                refresh_btn.click(
                    fn=lambda: create_analytics_dashboard([]),
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
    app.launch(server_name="0.0.0.0", server_port=7860)
