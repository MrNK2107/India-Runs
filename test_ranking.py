#!/usr/bin/env python3
"""Test ranking quality across multiple queries and analyze results."""
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents.executor import ExecutorAgent
from src.agents.orchestrator import _parse_query_text
from src.core.config import DATA_DIR
from src.core.models import Profile
from src.core.profile_store import ProfileStore
from src.language.multilingual import MultilingualEmbedder
from src.matching.scorer import CandidateScorer
from src.matching.behavioral_scorer import (
    compute_behavioral_score,
    compute_career_trajectory,
    compute_skill_proficiency,
    detect_honeypot,
)
from src.search.bm25_search import BM25Search
from src.search.hybrid import HybridSearch
from src.search.reranker import CrossEncoderReranker
from src.search.vector_search import VectorSearch


TEST_QUERIES = [
    "senior software engineer python java javascript aws postgresql",
    "data scientist machine learning python pytorch sql nlp",
    "full stack developer react node.js python typescript",
    "devops engineer docker kubernetes terraform aws ci/cd",
]

# Company tier maps
TIER1 = {"google", "microsoft", "amazon", "meta", "apple", "netflix",
         "stripe", "atlassian", "twitter", "linkedin", "uber", "airbnb",
         "flipkart", "swiggy", "zomato", "razorpay", "cred", "ola",
         "bytedance", "phonepe", "groww", "upstox", "zerodha"}
TIER2 = {"infosys", "tcs", "wipro", "hcl", "tech mahindra", "cognizant",
         "accenture", "capgemini", "l&t infotech", "mindtree", "mphasis",
         "oracle", "ibm", "sap", "salesforce", "vmware", "cisco",
         "dell", "hp", "adobe", "paypal", "intuit"}


def _calc_months(start, end):
    if not start or not end:
        return None
    try:
        sp = str(start).split("-")
        ep = str(end).split("-")
        if len(sp) < 2 or len(ep) < 2:
            return None
        return (int(ep[0]) - int(sp[0])) * 12 + (int(ep[1]) - int(sp[1]))
    except (ValueError, IndexError):
        return None


def get_job_hopping(candidate: Profile):
    """Calculate avg tenure."""
    tenures = []
    for job in candidate.experience:
        m = _calc_months(job.start_date, job.end_date)
        if m and m > 0:
            tenures.append(m)
    if tenures and len(candidate.experience) >= 3:
        return sum(tenures) / len(tenures)
    return None


def profile_summary(candidate: Profile, pid: str):
    """Return a detailed profile summary for analysis."""
    p = candidate.professional
    sig = candidate.signals

    # Experience list
    career = []
    for job in sorted(candidate.experience, key=lambda e: str(e.start_date or "")):
        tenure = _calc_months(job.start_date, job.end_date)
        tenure_str = f"{tenure // 12}y{tenure % 12}m" if tenure else "?"
        career.append(f"  {job.title} @ {job.company} ({tenure_str})")

    skills_detail = []
    for s in candidate.skills[:8]:
        skills_detail.append(f"{s.name}({s.proficiency or '?'}, {s.years_used or '?'}y)")

    hp = detect_honeypot(candidate)
    bh = f"{compute_behavioral_score(sig) * 100:.0f}%"
    ct = f"{compute_career_trajectory(candidate) * 100:.0f}%"
    jh = get_job_hopping(candidate)
    jh_str = f"{jh:.0f}m" if jh else "N/A"

    signals_str = []
    if sig.saved_by_recruiters_30d: signals_str.append(f"saved={sig.saved_by_recruiters_30d}")
    if sig.recruiter_response_rate: signals_str.append(f"resp={sig.recruiter_response_rate:.0%}")
    if sig.profile_completeness_score: signals_str.append(f"complete={sig.profile_completeness_score:.0f}%")
    if sig.open_to_work: signals_str.append("open2work")
    if sig.notice_period_days and sig.notice_period_days <= 30: signals_str.append("immediate")
    if sig.verified_email and sig.verified_phone: signals_str.append("verified")

    return {
        "name": candidate.personal.name if candidate.personal else "?",
        "title": p.current_title if p else "N/A",
        "company": p.current_company if p else "N/A",
        "exp_years": p.total_experience_years if p else 0,
        "exp_count": len(candidate.experience),
        "city": candidate.personal.location.city if candidate.personal and candidate.personal.location else "?",
        "skills": skills_detail,
        "skills_count": len(candidate.skills),
        "career": career,
        "beh_score": bh,
        "car_traj": ct,
        "job_hop": jh_str,
        "signals": ", ".join(signals_str) if signals_str else "none",
        "honeypot": "YES" if hp else "no",
        "honeypot_reason": hp or "",
        "company_tier": "tier1" if (p and p.current_company and p.current_company.lower() in TIER1)
                        else ("tier2" if (p and p.current_company and p.current_company.lower() in TIER2) else "other"),
    }


async def analyze_query(query: str, executor: ExecutorAgent, profiles: ProfileStore):
    """Run a query and analyze the ranking quality."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")

    parsed = _parse_query_text(query)
    results = await executor.execute(parsed, top_k=50)

    print(f"Total results: {len(results)}")
    print(f"\n--- TOP 10 ---")
    for r in results[:10]:
        p = profiles.get(r.profile_id)
        if not p:
            continue
        info = profile_summary(p, r.profile_id)
        print(f"\n  #{r.rank} {r.profile_id} | score={r.scores.overall:.4f}")
        print(f"    {info['name']} — {info['title']} @ {info['company']}")
        print(f"    Skills: {', '.join(info['skills'][:6])}")
        if r.matched_skills:
            print(f"    Matched: {', '.join(r.matched_skills[:6])}")
        print(f"    Exp: {info['exp_years']}y across {info['exp_count']} roles | City: {info['city']}")
        print(f"    Career_traj: {info['car_traj']} | Job_hop: {info['job_hop']} | Beh: {info['beh_score']} | Tier: {info['company_tier']}")
        print(f"    Signals: {info['signals']}")
        print(f"    Honeypot: {info['honeypot']} {info['honeypot_reason']}")
        print(f"    Score breakdown: cross_enc={r.scores.cross_encoder_score or 0:.3f} "
              f"sem={r.scores.semantic_similarity:.3f} kw={r.scores.keyword_match:.3f} "
              f"skill={r.scores.skill_match:.3f} exp={r.scores.experience_match:.3f} "
              f"beh={r.scores.behavioral_score or 0:.3f} car={r.scores.career_trajectory_score or 0:.3f} "
              f"prof={r.scores.skill_proficiency_score or 0:.3f}")

    print(f"\n--- BOTTOM 5 ---")
    for r in results[-5:]:
        p = profiles.get(r.profile_id)
        if not p:
            continue
        info = profile_summary(p, r.profile_id)
        print(f"\n  #{r.rank} {r.profile_id} | score={r.scores.overall:.4f}")
        print(f"    {info['name']} — {info['title']} @ {info['company']}")
        print(f"    Skills: {', '.join(info['skills'][:4])}")
        print(f"    Exp: {info['exp_years']}y | Career: {info['car_traj']} | Beh: {info['beh_score']} | Honeypot: {info['honeypot']}")

    # Quality analysis
    print(f"\n--- QUALITY ANALYSIS ---")
    issues = []

    # Check top 3 — do they LOOK like good candidates for this query?
    for i, r in enumerate(results[:3], 1):
        p = profiles.get(r.profile_id)
        if not p:
            continue
        info = profile_summary(p, r.profile_id)
        # Check relevance
        query_skills = set(w.lower() for w in query.split() if len(w) > 2)
        candidate_text = (p.raw_text + " " + " ".join(s.name for s in p.skills)).lower()
        matched_query = [s for s in query_skills if s in candidate_text]
        if len(matched_query) < len(query_skills) * 0.4:
            issues.append(f"  #{r.rank} {r.profile_id}: Only {len(matched_query)}/{len(query_skills)} query terms found in profile — possible false positive")
        if info['honeypot'] == "YES":
            issues.append(f"  #{r.rank} {r.profile_id}: HONEYPOT in top results! ({info['honeypot_reason']})")

    # Check bottom 5 — should be genuinely worse
    for r in results[-5:]:
        p = profiles.get(r.profile_id)
        if not p:
            continue
        info = profile_summary(p, r.profile_id)
        if info['honeypot'] != "YES" and info['company_tier'] == "tier1":
            issues.append(f"  WARNING: tier1 candidate {r.profile_id} at bottom (score={r.scores.overall:.3f}) — might be unfair")

    # Check for score inversions (non-monotonic)
    for i in range(len(results) - 1):
        if results[i].scores.overall < results[i + 1].scores.overall:
            issues.append(f"  SCORE INVERSION: #{results[i].rank} ({results[i].scores.overall:.4f}) < #{results[i+1].rank} ({results[i+1].scores.overall:.4f})")

    if issues:
        print("Issues found:")
        for iss in issues:
            print(f"  {iss}")
    else:
        print("No issues detected — ranking looks correct.")

    return results


async def main():
    print("Loading search system...", file=sys.stderr)

    indexes_dir = DATA_DIR / "indexes"
    embedder = MultilingualEmbedder()
    _ = embedder.model

    vector_search = VectorSearch()
    vector_search.load(indexes_dir / "faiss_index.bin", indexes_dir / "faiss_id_map.json")
    bm25_search = BM25Search()
    bm25_search.load(indexes_dir / "bm25_index.pkl")
    hybrid_search = HybridSearch(vector_search, bm25_search, embedder)
    reranker = CrossEncoderReranker(timeout_ms=0)
    scorer = CandidateScorer()
    profiles = ProfileStore()
    profiles.load_sample(DATA_DIR / "samples" / "sample_candidates.json")
    executor = ExecutorAgent(hybrid_search, reranker, scorer, profiles)

    for query in TEST_QUERIES:
        await analyze_query(query, executor, profiles)

    # Also test the full submission pipeline
    print(f"\n{'='*80}")
    print(f"RUNNING FULL SUBMISSION PIPELINE")
    print(f"{'='*80}")

    # Now do a comprehensive analysis of top-10 and bottom-10 in submission.csv
    print(f"\n\nFINAL SUBMISSION ANALYSIS")
    print(f"{'='*80}")

    all_profiles_map = profiles.get_all_sample()
    final_candidates = []

    for query in [
        "senior software engineer python java javascript aws postgresql distributed systems microservices",
        "backend developer python django fastapi aws postgresql redis docker kafka",
        "frontend developer react typescript next.js javascript css tailwind html",
        "full stack developer react node.js python typescript mongodb next.js aws",
        "senior data scientist machine learning python pytorch sql nlp deep learning analytics",
        "data engineer apache spark airflow kafka python etl bigquery snowflake aws",
        "ml engineer deep learning computer vision nlp pytorch tensorflow mloops python",
        "senior devops engineer docker kubernetes terraform aws ci/cd argocd prometheus",
        "cloud solutions architect aws azure gcp terraform kubernetes system design",
        "senior java developer spring boot microservices hibernate kafka restful api mysql",
        "mobile developer android kotlin flutter ios swift react native dart",
        "senior python developer fastapi flask django postgresql redis docker aws",
        "engineering manager tech lead distributed systems scala go leadership system design",
        "product manager analytics saas b2b agile strategy stakeholder management",
        "cybersecurity engineer application security penetration testing cloud security python",
        "qa automation engineer selenium cypress pytest playwright ci/cd python js",
        "solutions architect system design scalability microservices cloud distributed systems",
    ]:
        try:
            parsed = _parse_query_text(query)
            results = await executor.execute(parsed, top_k=50)
            for r in results:
                pid = r.profile_id
                existing = [c for c in final_candidates if c["pid"] == pid]
                if existing:
                    if r.scores.overall > existing[0]["score"]:
                        existing[0]["score"] = r.scores.overall
                        existing[0]["result"] = r
                else:
                    final_candidates.append({"pid": pid, "score": r.scores.overall, "result": r})
        except Exception as e:
            print(f"  Query failed: {e}", file=sys.stderr)

    final_candidates.sort(key=lambda x: (-x["score"], x["pid"]))

    print(f"\n--- TOP 10 in FINAL SUBMISSION ---")
    for i, c in enumerate(final_candidates[:10], 1):
        p = profiles.get(c["pid"])
        if not p:
            continue
        info = profile_summary(p, c["pid"])
        r = c["result"]
        print(f"\n  #{i} {c['pid']} | score={c['score']:.4f}")
        print(f"    {info['name']} — {info['title']} @ {info['company']}")
        print(f"    Skills: {', '.join(info['skills'][:6])}")
        if r.matched_skills:
            print(f"    Matched: {', '.join(r.matched_skills[:6])}")
        print(f"    Exp: {info['exp_years']}y across {info['exp_count']} roles")
        print(f"    Career_traj: {info['car_traj']} | Job_hop: {info['job_hop']} | Beh: {info['beh_score']} | Tier: {info['company_tier']}")
        print(f"    Signals: {info['signals']}")
        print(f"    Honeypot: {info['honeypot']}")
        print(f"    BREAKDOWN: cross_enc={r.scores.cross_encoder_score or 0:.3f} "
              f"sem={r.scores.semantic_similarity:.3f} kw={r.scores.keyword_match:.3f} "
              f"skill={r.scores.skill_match:.3f} exp={r.scores.experience_match:.3f} "
              f"beh={r.scores.behavioral_score or 0:.3f} car={r.scores.career_trajectory_score or 0:.3f} "
              f"prof={r.scores.skill_proficiency_score or 0:.3f}")

    print(f"\n--- BOTTOM 10 in FINAL SUBMISSION ---")
    for i, c in enumerate(final_candidates[-10:], len(final_candidates) - 9):
        p = profiles.get(c["pid"])
        if not p:
            continue
        info = profile_summary(p, c["pid"])
        r = c["result"]
        print(f"\n  #{i} {c['pid']} | score={c['score']:.4f}")
        print(f"    {info['name']} — {info['title']} @ {info['company']}")
        print(f"    Skills: {', '.join(info['skills'][:4])}")
        print(f"    Exp: {info['exp_years']}y | Honeypot: {info['honeypot']} {info['honeypot_reason'][:50]}")
        print(f"    Beh: {info['beh_score']} | Car: {info['car_traj']} | Job_hop: {info['job_hop']}")
        print(f"    BREAKDOWN: cross_enc={r.scores.cross_encoder_score or 0:.3f} "
              f"sem={r.scores.semantic_similarity:.3f} kw={r.scores.keyword_match:.3f} "
              f"skill={r.scores.skill_match:.3f} exp={r.scores.experience_match:.3f} "
              f"beh={r.scores.behavioral_score or 0:.3f} car={r.scores.career_trajectory_score or 0:.3f} "
              f"prof={r.scores.skill_proficiency_score or 0:.3f}")

    print(f"\n{'='*80}")
    print("ANALYSIS: Score range, distribution, and edge cases")
    print(f"{'='*80}")

    scores = [c["score"] for c in final_candidates]
    if scores:
        print(f"  Total candidates: {len(scores)}")
        print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"  Average: {sum(scores)/len(scores):.4f}")
        print(f"  Median: {sorted(scores)[len(scores)//2]:.4f}")
        print(f"  Top-10 avg: {sum(scores[:10])/10:.4f}")
        print(f"  Bottom-10 avg: {sum(scores[-10:])/10:.4f}")

        # Honeypot positions
        print(f"\n--- HONEYPOT PROFILES (penalized to bottom) ---")
        for i, c in enumerate(final_candidates, 1):
            p = profiles.get(c["pid"])
            if p and detect_honeypot(p):
                print(f"  #{i} {c['pid']} score={c['score']:.4f}")

    print(f"\n{'='*80}")
    print("RECOMMENDED FIXES (if any)")
    print(f"{'='*80}")

    # Final quality checks
    qa_issues = []
    for i, c in enumerate(final_candidates[:5], 1):
        p = profiles.get(c["pid"])
        if not p:
            continue
        info = profile_summary(p, c["pid"])
        if info['honeypot'] == "YES":
            qa_issues.append(f"  HONEYPOT in top 5! #{i} {c['pid']}")
        if not c["result"].matched_skills:
            qa_issues.append(f"  #{i} {c['pid']}: Top-ranked but 0 matched skills")

    # Check for duplicates
    pids = [c["pid"] for c in final_candidates]
    dupes = {pid for pid in pids if pids.count(pid) > 1}
    if dupes:
        qa_issues.append(f"  DUPLICATE CANDIDATES: {dupes}")

    # Check monotonic scores
    for i in range(len(final_candidates) - 1):
        if final_candidates[i]["score"] < final_candidates[i + 1]["score"]:
            qa_issues.append(f"  Score inversion at rank {i+1}")

    if qa_issues:
        for q in qa_issues:
            print(f"  {q}")
    else:
        print("  All quality checks passed — ranking is clean.")


if __name__ == "__main__":
    asyncio.run(main())
