"""Full pipeline end-to-end test with real Ollama LLM.
Tests every agent node in the LangGraph search pipeline."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("e2e-test")

errors: list[str] = []
successes: list[str] = []

def check(step: str, ok: bool, detail: str = ""):
    if ok:
        successes.append(step)
        logger.info(f"  PASS [{step}] {detail}")
    else:
        errors.append(step)
        logger.error(f"  FAIL [{step}] {detail}")

import traceback


async def run_tests():
    from src.core.models import MatchResult, MatchScores

    def _make_match_result(pid: str, name: str, title: str) -> MatchResult:
        """Build a minimal MatchResult for testing."""
        return MatchResult(
            query_id="e2e-test",
            rank=0,
            profile_id=pid,
            name=name,
            current_title=title,
            current_company="",
            matched_skills=[],
            missing_skills=[],
            scores=MatchScores(
                semantic_similarity=0.5,
                keyword_match=0.5,
                skill_match=0.5,
                experience_match=0.5,
                location_match=0.5,
                overall=0.5,
                confidence=0.5,
            ),
        )

    # ==========================================================
    # STEP 1: Config & LLM connectivity
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Config & LLM Connectivity")
    logger.info("=" * 60)

    from src.core.config import get_settings, get_llm_client

    settings = get_settings()
    check("config-loads", settings.llm_provider == "ollama", f"provider={settings.llm_provider}")

    client = get_llm_client()
    try:
        resp = await client.ainvoke("Say exactly: HELLO OLLAMA")
        text = resp.content if hasattr(resp, 'content') else str(resp)
        check("ollama-connect", "HELLO OLLAMA" in text.upper(), f"response: {text[:100]}")
    except Exception as e:
        check("ollama-connect", False, str(e))

    # ==========================================================
    # STEP 2: Profile Store & Index Loading
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Profile Store & Index Loading")
    logger.info("=" * 60)

    from src.core.profile_store import ProfileStore
    from src.core.config import DATA_DIR
    from src.core.constants import SAMPLE_PATH
    from src.matching.scorer import CandidateScorer, DEFAULT_SLIDER_WEIGHTS
    from src.matching.skill_matcher import SkillMatcher

    store = ProfileStore()
    check("profile-store-loaded", len(store) > 0, f"profiles={len(store)}")

    # Fetch a sample profile using get_all_sample
    samples = store.get_all_sample()
    check("sample-profiles-loaded", len(samples) > 0, f"count={len(samples)}")
    sample = next(iter(samples.values())) if samples else None
    check("profile-fetch", sample is not None, f"first profile id={sample.profile_id if sample else 'N/A'}")

    # Load indexes
    from src.search.vector_search import VectorSearch
    from src.search.bm25_search import BM25Search
    from src.search.hybrid import HybridSearch
    from src.search.reranker import CrossEncoderReranker
    from src.language.multilingual import MultilingualEmbedder

    index_dir = DATA_DIR / "indexes"
    faiss_path = index_dir / "faiss_index.bin"
    bm25_path = index_dir / "bm25_index.pkl"
    id_map_path = index_dir / "faiss_id_map.json"

    check("faiss-exists", faiss_path.exists(), str(faiss_path))
    check("bm25-exists", bm25_path.exists(), str(bm25_path))

    vector_search = VectorSearch(dimension=384)
    try:
        vector_search.load(faiss_path, id_map_path)
        check("faiss-load", vector_search.size > 0, f"vectors={vector_search.size}")
    except Exception as e:
        check("faiss-load", False, str(e))

    bm25 = BM25Search()
    try:
        bm25.load(bm25_path)
        check("bm25-load", bm25.size > 0, f"docs={bm25.size}")
    except Exception as e:
        check("bm25-load", False, str(e))

    logger.info("Loading embedding model (may take ~30s on first run)...")
    embedder = MultilingualEmbedder()
    hybrid = HybridSearch(vector_search, bm25, embedder)
    reranker = CrossEncoderReranker(timeout_ms=5000)

    # Test search
    try:
        results = hybrid.search("python developer", top_k=5)
        check("hybrid-search", len(results) > 0, f"results={len(results)}")
    except Exception as e:
        check("hybrid-search", False, str(e))

    # Test empty search
    try:
        empty_results = hybrid.search("zzzzzzzzzznonexistent", top_k=5)
        check("hybrid-empty-search", len(empty_results) > 0, "empty query still returns results as fallback")
    except Exception as e:
        check("hybrid-empty-search", False, str(e))

    # ==========================================================
    # STEP 3: Create executor with dependencies
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Create agent dependencies")
    logger.info("=" * 60)

    scorer = CandidateScorer()
    from src.agents.executor import ExecutorAgent
    executor = ExecutorAgent(hybrid, reranker, scorer, store)

    from src.agents.planner import PlannerAgent
    planner = PlannerAgent()

    from src.agents.reflector import ReflectorAgent
    reflector = ReflectorAgent()

    # ==========================================================
    # STEP 4: Planner Agent (async)
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 4: Planner Agent")
    logger.info("=" * 60)

    try:
        query = "senior DevOps engineer with 5+ years AWS experience in Bangalore"
        plan = await planner.plan(query)
        check("planner-plan", plan is not None, f"plan type={type(plan).__name__ if plan else 'None'}")
        if plan:
            req_skills = plan.required_skills
            check("planner-has-required-skills", len(req_skills) > 0,
                  f"required_skills={[str(s) for s in req_skills[:5]]}")
            check("planner-has-experience", plan.experience is not None,
                  f"exp={plan.experience}")
    except Exception as e:
        check("planner-plan", False, f"Exception: {e}")
        traceback.print_exc()

    # ==========================================================
    # STEP 5: Executor Agent (async)
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 5: Executor Agent")
    logger.info("=" * 60)

    try:
        exec_results = await executor.execute(plan, top_k=5)
        check("executor-execute", exec_results is not None, f"type={type(exec_results).__name__}")
        if exec_results and isinstance(exec_results, list):
            check("executor-has-results", len(exec_results) > 0, f"count={len(exec_results)}")
            if exec_results:
                r = exec_results[0]
                check("executor-result-has-score", hasattr(r, 'scores') and r.scores is not None,
                      f"scores={r.scores}")
    except Exception as e:
        check("executor-execute", False, f"Exception: {e}")
        traceback.print_exc()

    # ==========================================================
    # STEP 6: Scorer with slider weights
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 6: Scorer with Slider Weights")
    logger.info("=" * 60)

    slider_weights = {
        "skill_match": 80,
        "experience_match": 70,
        "education_match": 50,
        "assessment_score": 40,
        "behavioral_signals": 30,
        "cultural_fit": 20,
    }
    try:
        scores = scorer.compute_overall(
            {
                "semantic_similarity": 0.85,
                "keyword_match": 0.75,
                "skill_match": 0.90,
                "experience_match": 0.80,
                "location_match": 0.60,
                "education_match": 0.70,
                "cross_encoder_score": 0.50,
                "behavioral_signals": 0.65,
                "cultural_fit": 0.55,
            },
            slider_weights=slider_weights,
        )
        check("scorer-overall", scores.overall > 0, f"overall={scores.overall:.3f}")
        check("scorer-confidence", scores.confidence > 0, f"confidence={scores.confidence:.3f}")
    except Exception as e:
        check("scorer-overall", False, f"Exception: {e}")
        traceback.print_exc()

    # ==========================================================
    # STEP 7: Reflector Agent
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 7: Reflector Agent")
    logger.info("=" * 60)

    # Build a MatchResult list for the reflector
    from src.core.models import MatchResult, MatchScores
    match_results = []
    if exec_results and isinstance(exec_results, list):
        for mr in exec_results:
            match_results.append(mr)
    try:
        eval_results = await reflector.reflect(plan, match_results)
        check("reflector-evaluate", eval_results is not None,
              f"type={type(eval_results).__name__ if eval_results else 'None'}")
        if eval_results and isinstance(eval_results, dict):
            check("reflector-has-feedback", "feedback" in eval_results or "evaluations" in eval_results,
                  f"keys={list(eval_results.keys())}")
    except Exception as e:
        check("reflector-evaluate", False, f"Exception: {e}")
        traceback.print_exc()

    # ==========================================================
    # STEP 8: Rationale Generator
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 8: Rationale Generator")
    logger.info("=" * 60)

    from src.rationale.generator import RationaleGenerator

    gen = RationaleGenerator()
    try:
        # Use template rationale with a sample match result
        if exec_results and isinstance(exec_results, list) and len(exec_results) > 0:
            mr = exec_results[0]
            profile = store.get(mr.profile_id)
            if profile:
                rationale = gen._template_rationale(mr, profile)
                check("rationale-template-fallback", rationale is not None,
                      f"len={len(rationale.generated_rationale) if rationale else 0}")
            else:
                check("rationale-template-fallback", True, "skipped (no profile)")
        else:
            check("rationale-template-fallback", True, "skipped (no results)")
    except Exception as e:
        check("rationale-template-fallback", False, str(e))
        traceback.print_exc()

    # ==========================================================
    # STEP 9: Full Orchestrator
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 9: Full Orchestrator (LangGraph)")
    logger.info("=" * 60)

    from src.agents.orchestrator import Orchestrator

    orchestrator = Orchestrator(planner, executor, reflector)
    try:
        result = await orchestrator.run(
            raw_query="senior software engineer Python AWS",
            slider_weights=DEFAULT_SLIDER_WEIGHTS,
        )
        check("orchestrator-run", result is not None, f"type={type(result).__name__ if result else 'None'}")
        if result:
            from src.core.models import SearchResponse
            if isinstance(result, SearchResponse):
                results_arr = result.results
                check("orchestrator-has-results", len(results_arr) > 0, f"count={len(results_arr)}")
                search_metadata = result.search_metadata
                check("orchestrator-has-metadata", search_metadata is not None,
                      f"metadata={search_metadata}")
                if search_metadata:
                    check("orchestrator-processing-time", search_metadata.total_time_ms > 0,
                          f"time={search_metadata.total_time_ms}ms")
                    check("orchestrator-listwise",
                          search_metadata.listwise_ranked is not None,
                          f"listwise_ranked={search_metadata.listwise_ranked}")
    except Exception as e:
        check("orchestrator-run", False, f"Exception: {e}")
        traceback.print_exc()

    # ==========================================================
    # STEP 10: Fairness Components
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 10: Fairness Components")
    logger.info("=" * 60)

    from src.fairness.anonymizer import anonymize_text_for_bias, anonymize_profile, style_anonymize
    from src.fairness.bias_detector import BiasDetector

    try:
        pii_text = "John Doe worked at Google in Bangalore. Email: johndoe@email.com"
        anon_text = anonymize_text_for_bias(pii_text)
        check("pii-anonymize-text", "[NAME]" in anon_text, f"result: {anon_text[:80]}")
    except Exception as e:
        check("pii-anonymize-text", False, str(e))

    try:
        styled = style_anonymize("I spearheaded the devops transformation and leveraged cutting-edge tech")
        check("style-anonymize", "spearheaded" not in styled and "leveraged" not in styled,
              f"result: {styled[:80]}")
    except Exception as e:
        check("style-anonymize", False, str(e))

    detector = BiasDetector()
    try:
        if samples:
            sample = next(iter(samples.values()))
            mr = _make_match_result(
                pid=sample.profile_id,
                name=sample.personal.name if sample.personal else sample.profile_id,
                title=sample.professional.current_title if sample.professional else "",
            )
            profiles_dict = {sample.profile_id: sample}
            bias_result = detector.detect_bias(
                matches=[mr], profiles=profiles_dict, bias_type="name",
            )
            check("bias-detector", isinstance(bias_result, dict), f"result={bias_result}")
    except Exception as e:
        check("bias-detector", False, str(e))

    # ==========================================================
    # STEP 11: Listwise Ranking
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 11: Plackett-Luce Listwise Ranking")
    logger.info("=" * 60)

    from src.ranking.listwise_ranker import PlackettLuceRanker
    from src.core.models import MatchResult, MatchScores

    ranker = PlackettLuceRanker()
    try:
        # Build MatchResult objects from sample profiles
        results_for_rank: list[MatchResult] = []
        if samples:
            for pid, prof in list(samples.items())[:10]:
                mr = _make_match_result(
                    pid=pid,
                    name=prof.personal.name if prof.personal else pid,
                    title=prof.professional.current_title if prof.professional else "",
                )
                results_for_rank.append(mr)
        if results_for_rank:
            ranked = await ranker.arank(results_for_rank)
            check("listwise-rank", ranked is not None and len(ranked) > 0,
                  f"count={len(ranked) if ranked else 0}")
    except Exception as e:
        check("listwise-rank", False, f"Exception: {e}")
        traceback.print_exc()

    # ==========================================================
    # STEP 12: Evaluation Metrics
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 12: Evaluation Metrics")
    logger.info("=" * 60)

    from src.evaluation.metrics import precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k

    try:
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}
        p_at_3 = precision_at_k(retrieved, relevant, 3)
        r_at_5 = recall_at_k(retrieved, relevant, 5)
        mrr = mean_reciprocal_rank(retrieved, relevant)
        ndcg = ndcg_at_k(retrieved, relevant, 5)
        check("eval-precision", 0.0 <= p_at_3 <= 1.0, f"p@3={p_at_3:.3f}")
        check("eval-recall", 0.0 <= r_at_5 <= 1.0, f"r@5={r_at_5:.3f}")
        check("eval-mrr", 0.0 <= mrr <= 1.0, f"mrr={mrr:.3f}")
        check("eval-ndcg", 0.0 <= ndcg <= 1.0, f"ndcg={ndcg:.3f}")
    except Exception as e:
        check("eval-metrics", False, str(e))

    # ==========================================================
    # STEP 13: Hinglish query
    # ==========================================================
    logger.info("=" * 60)
    logger.info("STEP 13: Hinglish Query")
    logger.info("=" * 60)

    from src.language.code_mixed import CodeMixedProcessor

    cm_processor = CodeMixedProcessor()
    try:
        hinglish_query = "Mujhe ek senior Python developer chahiye with 5 years experience"
        lang_result = cm_processor.detect_code_mixed(hinglish_query)
        check("hinglish-detect", lang_result is not None, f"result={lang_result}")
    except Exception as e:
        check("hinglish-detect", False, str(e))

    try:
        hinglish_plan = await planner.plan(hinglish_query)
        check("hinglish-plan", hinglish_plan is not None,
              f"plan type={type(hinglish_plan).__name__ if hinglish_plan else 'None'}")
    except Exception as e:
        check("hinglish-plan", False, str(e))

    # ==========================================================
    # SUMMARY
    # ==========================================================
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total steps: {len(successes) + len(errors)}")
    logger.info(f"Passed: {len(successes)}")
    logger.info(f"Failed: {len(errors)}")

    if errors:
        logger.error("FAILED STEPS:")
        for e in errors:
            logger.error(f"  - {e}")
        return 1
    else:
        logger.info("ALL STEPS PASSED!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)
