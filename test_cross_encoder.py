#!/usr/bin/env python3
"""Test cross-encoder directly on various profile/text pairs."""
import os, sys
sys.path.insert(0, '/home/nanda/India-Runs')
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.search.reranker import CrossEncoderReranker
from src.core.profile_store import ProfileStore
from src.core.config import DATA_DIR

reranker = CrossEncoderReranker(timeout_ms=0)

profiles = ProfileStore()
profiles.load_sample(DATA_DIR / "samples" / "sample_candidates.json")

test_cases = [
    ("senior software engineer python java aws", "CAND_0000088", "Riya - Software Engineer @ TCS (Java, JavaScript, AWS, Azure)"),
    ("senior software engineer python java aws", "CAND_0000100", "Amit - Data Scientist @ Microsoft (Python, PostgreSQL, AWS, Kotlin)"),
    ("senior software engineer python java aws", "CAND_0000025", "Anika - Frontend Engineer @ TechM (JS, TS, GCP)"),
    ("senior software engineer python java aws", "CAND_0000044", "Vihaan - Frontend Engineer @ TechM (JS, Python, Hadoop)"),
    ("data scientist machine learning python pytorch", "CAND_0000001", "Ira - Backend Engineer @ Mindtree (NLP, Image Class, LLMs)"),
    ("data scientist machine learning python pytorch", "CAND_0000100", "Amit - Data Scientist @ Microsoft (Python, PostgreSQL)"),
    ("full stack developer react node.js python", "CAND_0000044", "Vihaan - Frontend @ TechM (JS, Python)"),
    ("full stack developer react node.js python", "CAND_0000065", "Rajesh - Data Scientist @ Google (React, Django)"),
    ("full stack developer react node.js python", "CAND_0000014", "Atharv - Frontend @ Zomato (React, BigQuery, OpenCV)"),
    ("software engineer with experience", "CAND_0000051", "Deepak - Engineering Manager @ Swiggy (14.8y, CI/CD, Java) - NO career_history"),
]

print("CROSS-ENCODER TEST RESULTS")
print("="*80)
for query, pid, description in test_cases:
    p = profiles.get(pid)
    if not p:
        print(f"{pid}: NOT FOUND")
        continue
    
    # Use score_pair directly
    score = reranker.score_pair(query, p.raw_text)
    
    print(f"\nQuery: {query}")
    print(f"PID:   {pid} — {description}")
    print(f"Text:  {p.raw_text[:150]}...")
    print(f"Score: {score if score is not None else 'N/A'}")
    print()

# Also check the relative ordering
print("="*80)
print("RELATIVE ORDERING TEST: query='senior software engineer python java aws'")
print("="*80)

query = "senior software engineer python java aws"
results = []
for pid in ["CAND_0000088", "CAND_0000100", "CAND_0000025", "CAND_0000044", "CAND_0000065", "CAND_0000055", "CAND_0000057"]:
    p = profiles.get(pid)
    if p:
        score = reranker.score_pair(query, p.raw_text)
        p_info = f"{p.personal.name} — {p.professional.current_title} @ {p.professional.current_company}"
        results.append((score, pid, p_info))

results.sort(key=lambda x: -x[0])
for i, (score, pid, info) in enumerate(results, 1):
    print(f"  #{i} {pid}: score={score:.4f} — {info}")
