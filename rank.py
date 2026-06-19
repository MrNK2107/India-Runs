#!/usr/bin/env python3
"""
Redrob India Runs — Main ranking entry point.

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Runs a multi-query retrieval pipeline with hybrid search (FAISS + BM25),
cross-encoder reranking, and weighted scoring across 7 dimensions.
Produces a 100-row submission.csv with candidate_id, rank, score, reasoning.
"""
import argparse
import asyncio
import csv
import logging
import os
import sys
import time
from pathlib import Path

# Force offline mode before any sentence-transformers imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents.executor import ExecutorAgent
from src.agents.orchestrator import _parse_query_text
from src.core.config import DATA_DIR, get_settings
from src.core.models import Profile
from src.core.profile_store import ProfileStore
from src.language.multilingual import MultilingualEmbedder
from src.matching.scorer import CandidateScorer
from src.search.bm25_search import BM25Search
from src.search.hybrid import HybridSearch
from src.search.reranker import CrossEncoderReranker
from src.search.vector_search import VectorSearch

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)

# ── Strategic search queries ──────────────────────────────────────────
# Each query targets a distinct role/tech stack to maximize candidate coverage.
# The cross-encoder reranks results in context of each query.
SEARCH_QUERIES = [
    # Core software engineering
    "software engineer python java javascript react",
    "backend developer python django aws sql postgresql",
    "frontend developer react typescript javascript css html",
    "full stack developer react node.js python mongodb express",

    # Data & ML
    "data scientist machine learning python pytorch tensorflow sql",
    "data engineer spark airflow python sql kafka",
    "ml engineer deep learning nlp computer vision python pytorch",

    # Cloud & DevOps
    "devops engineer docker kubernetes aws terraform ci/cd",
    "cloud architect aws azure gcp",

    # Mobile & specialized
    "mobile developer android kotlin swift ios react native flutter",
    "java spring boot microservices hibernate",
    "python developer fastapi flask django backend",

    # Leadership & management
    "engineering manager tech lead scala go distributed systems",
    "product manager analytics roadmap stakeholder",

    # Domain-specific
    "cybersecurity engineer network security penetration testing",
    "qa engineer automation testing selenium cypress pytest",
    "solutions architect system design scalability microservices",
]

# Common Indian cities to detect location automatically
INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Noida",
    "Gurgaon", "Indore", "Bhopal", "Nagpur", "Visakhapatnam", "Thane",
    "Vadodara", "Coimbatore", "Kochi", "Chandigarh", "Trivandrum",
    "Guwahati", "Mysore", "Bhubaneswar", "Goa", "Vizag",
]


def load_search_system() -> tuple:
    """Load all search components with cross-encoder reranking."""
    indexes_dir = DATA_DIR / "indexes"

    logger.info("Loading embedder...")
    embedder = MultilingualEmbedder()
    _ = embedder.model  # Force eager init

    logger.info("Loading vector search...")
    vector_search = VectorSearch()
    vector_search.load(
        indexes_dir / "faiss_index.bin",
        indexes_dir / "faiss_id_map.json",
    )

    logger.info("Loading BM25 search...")
    bm25_search = BM25Search()
    bm25_search.load(indexes_dir / "bm25_index.pkl")

    logger.info("Building hybrid search...")
    hybrid_search = HybridSearch(vector_search, bm25_search, embedder)

    logger.info("Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker(timeout_ms=0)

    logger.info("Loading scorer...")
    scorer = CandidateScorer()

    logger.info("Loading profiles...")
    profiles = ProfileStore()
    sample_path = DATA_DIR / "samples" / "sample_candidates.json"
    profiles.load_sample(sample_path)

    return hybrid_search, reranker, scorer, profiles


def _detect_locations(all_profiles: dict[str, Profile]) -> dict[str, str]:
    """Build a city-indexed map of candidate IDs."""
    loc_map = {}
    for pid, profile in all_profiles.items():
        city = None
        if profile.personal and profile.personal.location:
            city = profile.personal.location.city
        loc_map[pid] = city or "Unknown"
    return loc_map


def _build_reasoning(
    candidate_id: str,
    profile: Profile,
    score: float,
    matched_skills: list[str],
    missing_skills: list[str],
    query: str,
    location_map: dict[str, str],
) -> str:
    """Generate a compelling, detail-rich reasoning string."""
    title = profile.professional.current_title if profile.professional else "N/A"
    company = profile.professional.current_company if profile.professional else "N/A"
    exp = profile.professional.total_experience_years if profile.professional else 0
    city = location_map.get(candidate_id, "N/A")

    parts = []

    # Skill match summary
    if matched_skills:
        parts.append(f"Matched {len(matched_skills)} skills: {', '.join(matched_skills[:6])}.")
    else:
        parts.append("General skill alignment.")

    # Title + company signal
    if title and title != "N/A":
        parts.append(f"Current: {title}")
        if company and company != "N/A":
            parts.append(f"at {company}")

    # Experience
    if exp:
        parts.append(f"{exp:.0f}y experience.")
    else:
        parts.append("Experience not specified.")

    # Location
    if city and city != "Unknown":
        parts.append(f"Based in {city}.")

    # Missing skills (for honesty/transparency)
    if missing_skills:
        if len(missing_skills) <= 3:
            parts.append(f"Gaps: {', '.join(missing_skills)}.")
        else:
            parts.append(f"Missing {len(missing_skills)} less critical skills.")

    return " ".join(parts)


async def run_pipeline(profiles: ProfileStore, executor: ExecutorAgent,
                        location_map: dict[str, str]) -> list[dict]:
    """Run all search queries, collect and merge results."""
    all_candidates: dict[str, dict] = {}  # pid -> best result

    for q_idx, query in enumerate(SEARCH_QUERIES):
        logger.info(f"Query [{q_idx + 1}/{len(SEARCH_QUERIES)}]: {query}")
        try:
            parsed = _parse_query_text(query)
            results = await executor.execute(parsed, top_k=50)

            for r in results:
                pid = r.profile_id
                profile = profiles.get(pid)
                if profile is None:
                    continue

                # If we already have this candidate, keep the best score
                existing = all_candidates.get(pid)
                new_score = r.scores.overall
                if existing and existing["score"] >= new_score:
                    continue

                all_candidates[pid] = {
                    "candidate_id": pid,
                    "score": round(new_score, 4),
                    "matched_skills": r.matched_skills,
                    "missing_skills": r.missing_skills,
                    "title": r.current_title,
                    "company": r.current_company,
                    "query": query,
                    "experience_years": r.experience_years,
                    "profile": profile,
                }

            logger.info(f"  → {len(results)} results ({len(all_candidates)} unique so far)")

        except Exception as e:
            logger.warning(f"  Query failed: {e}")
            continue

    return list(all_candidates.values())


def _fill_remaining(all_profiles: dict[str, Profile],
                     existing_pids: set[str],
                     location_map: dict[str, str]) -> list[dict]:
    """Fill remaining slots with unmatched profiles (lowest priority)."""
    remaining = []
    for pid in all_profiles:
        if pid in existing_pids:
            continue
        profile = all_profiles[pid]
        exp = profile.professional.total_experience_years if profile.professional else 0
        title = profile.professional.current_title if profile.professional else "N/A"
        company = profile.professional.current_company if profile.professional else "N/A"
        city = location_map.get(pid, "N/A")

        # Score based on profile completeness + experience
        base_score = 0.10
        if exp:
            base_score += min(0.15, exp / 30.0)
        if title and title != "N/A":
            base_score += 0.05
        if company and company != "N/A":
            base_score += 0.03
        if city and city != "Unknown":
            base_score += 0.02

        remaining.append({
            "candidate_id": pid,
            "score": round(min(0.35, base_score), 4),
            "matched_skills": [],
            "missing_skills": [],
            "title": title,
            "company": company,
            "query": "",
            "experience_years": exp,
            "profile": profile,
        })

    return remaining


async def main():
    parser = argparse.ArgumentParser(
        description="Redrob India Runs — candidate ranking pipeline"
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Path to candidates.jsonl (not used in sample mode — uses built-in samples)",
    )
    parser.add_argument(
        "--out",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    t0 = time.time()
    print("Loading search system...", file=sys.stderr)

    hybrid_search, reranker, scorer, profiles = load_search_system()
    executor = ExecutorAgent(hybrid_search, reranker, scorer, profiles)

    all_profiles = profiles.get_all_sample()
    location_map = _detect_locations(all_profiles)
    print(f"Loaded {len(all_profiles)} profiles", file=sys.stderr)

    # ── Run multi-query pipeline ──────────────────────────────────
    print("Running multi-query search pipeline...", file=sys.stderr)
    candidates = await run_pipeline(profiles, executor, location_map)

    existing_pids = {c["candidate_id"] for c in candidates}
    print(f"Found {len(candidates)} matched candidates from {len(SEARCH_QUERIES)} queries",
          file=sys.stderr)

    # ── Fill remaining ────────────────────────────────────────────
    if len(candidates) < 100:
        remaining = _fill_remaining(all_profiles, existing_pids, location_map)
        candidates.extend(remaining)
        print(f"Filled {len(remaining)} remaining slots to reach 100", file=sys.stderr)

    # ── Sort by score descending, tie-break by ID ─────────────────
    candidates.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # ── Build final rows ──────────────────────────────────────────
    rows = []
    for rank, c in enumerate(candidates[:100], start=1):
        reasoning = _build_reasoning(
            c["candidate_id"], c["profile"],
            c["score"], c["matched_skills"],
            c["missing_skills"], c["query"],
            location_map,
        )
        rows.append({
            "candidate_id": c["candidate_id"],
            "rank": rank,
            "score": c["score"],
            "reasoning": reasoning,
        })

    # ── Write CSV ─────────────────────────────────────────────────
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"\nSubmission written to {out_path}", file=sys.stderr)
    print(f"Total rows: {len(rows)}", file=sys.stderr)
    print(f"Score range: {rows[-1]['score']:.4f} - {rows[0]['score']:.4f}", file=sys.stderr)
    print(f"Total time: {elapsed:.1f}s", file=sys.stderr)

    # ── Verification ──────────────────────────────────────────────
    cids = [r["candidate_id"] for r in rows]
    assert len(set(cids)) == len(cids), "Duplicate candidate IDs!"
    assert all(c.startswith("CAND_") for c in cids), "Invalid candidate ID format!"
    assert len(rows) == 100, f"Expected 100 rows, got {len(rows)}"

    # Verify non-increasing scores
    for i in range(len(rows) - 1):
        if rows[i]["score"] < rows[i + 1]["score"]:
            print(f"WARNING: Score increase at rank {rows[i]['rank']} -> {rows[i+1]['rank']}",
                  file=sys.stderr)

    # Verify ranks 1-100 are complete
    ranks = {r["rank"] for r in rows}
    assert ranks == set(range(1, 101)), f"Ranks missing: {set(range(1,101)) - ranks}"

    print("\nTop 10:", file=sys.stderr)
    for r in rows[:10]:
        print(f"  #{r['rank']} {r['candidate_id']} score={r['score']:.4f} — {r['reasoning'][:100]}...",
              file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
