from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.normalizer import normalize_redrob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_PATH = DATA_DIR / "samples" / "sample_candidates.json"
FULL_PATH = DATA_DIR / "profiles" / "candidates.jsonl"
QUERIES_DIR = DATA_DIR / "queries"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"

GT_COUNT = 500  # total queries to generate


def build_skill_clusters(
    profiles: list[tuple[str, set[str]]],
) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = {}
    for pid, skills in profiles:
        clusters[pid] = []
        for other_pid, other_skills in profiles:
            if other_pid == pid:
                continue
            overlap = len(skills & other_skills)
            union = len(skills | other_skills)
            jaccard = overlap / union if union > 0 else 0
            if jaccard >= 0.2:
                clusters[pid].append(other_pid)
    return clusters


def _load_sample_profiles() -> list[dict]:
    if not SAMPLE_PATH.exists():
        logger.warning(f"Sample file not found: {SAMPLE_PATH}")
        return []
    with open(SAMPLE_PATH) as f:
        return json.load(f)


def _load_full_profiles(sample_size: int = GT_COUNT) -> list[dict]:
    if not FULL_PATH.exists():
        logger.warning(f"Full profiles not found: {FULL_PATH}")
        return []
    import random
    random.seed(42)
    lines = FULL_PATH.read_text().strip().splitlines()
    sampled_lines = random.sample(lines, min(sample_size, len(lines)))
    result = []
    skipped = 0
    for line in sampled_lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    if skipped:
        logger.warning(f"Skipped {skipped} malformed lines")
    logger.info(f"Loaded {len(result)} profiles from full set")
    return result


def _make_query(raw: dict, normalized) -> tuple[str, str, str]:
    pid = normalized.profile_id
    headline = raw.get("profile", {}).get("headline", "")
    title = raw.get("profile", {}).get("current_title", "")
    company = raw.get("profile", {}).get("current_company", "")
    skill_names = {s.name for s in normalized.skills}

    top_skills = list(skill_names)[:5]
    parts = [title or headline] if title or headline else []
    parts.extend(top_skills)
    if company:
        parts.append(company)
    query_text = " ".join(parts) if parts else headline
    return pid, query_text, "en"


def generate() -> dict:
    raw_profiles = _load_sample_profiles()
    raw_profiles.extend(_load_full_profiles())

    queries: list[dict] = []
    ground_truth: dict[str, list[str]] = {}
    profiles_with_skills: list[tuple[str, set[str]]] = []

    for raw in raw_profiles[:GT_COUNT]:
        try:
            normalized = normalize_redrob(raw)
        except Exception:
            continue

        pid = normalized.profile_id
        skill_names = {s.name for s in normalized.skills}
        profiles_with_skills.append((pid, skill_names))

        pid, query_text, lang = _make_query(raw, normalized)

        qid = f"GT_{pid}"
        queries.append({
            "query_id": qid,
            "query": query_text,
            "language": lang,
            "source_profile": pid,
        })
        ground_truth[qid] = [pid]

    logger.info(f"Built {len(queries)} base queries from {len(raw_profiles)} profiles")

    # Efficient clustering: use skill-to-pid index to avoid O(n²)
    if len(profiles_with_skills) > 100:
        logger.info("Large dataset: using hash-based clustering")
        skill_set = {s for _, skills in profiles_with_skills for s in skills}
        skill_to_pids: dict[str, list[str]] = {s: [] for s in skill_set}
        for pid, skills in profiles_with_skills:
            for skill in skills:
                skill_to_pids[skill].append(pid)
        clusters: dict[str, list[str]] = {}
        for pid, skills in profiles_with_skills:
            related = set()
            for skill in skills:
                related.update(skill_to_pids.get(skill, []))
            related.discard(pid)
            clusters[pid] = list(related)
    else:
        clusters = build_skill_clusters(profiles_with_skills)

    for q in queries:
        qid = q["query_id"]
        pid = q["source_profile"]
        related = clusters.get(pid, [])
        seen = set(ground_truth.get(qid, []))
        for rpid in related:
            if rpid not in seen:
                ground_truth[qid].append(rpid)
                seen.add(rpid)

    total_relevant = sum(len(v) for v in ground_truth.values())
    avg_relevant = total_relevant / len(ground_truth) if ground_truth else 0
    logger.info(
        f"Ground truth: {len(ground_truth)} queries, "
        f"{total_relevant} total relevance labels, "
        f"{avg_relevant:.1f} avg per query"
    )

    QUERIES_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    queries_path = QUERIES_DIR / "queries.json"
    with open(queries_path, "w") as f:
        json.dump(queries, f, indent=2)
    logger.info(f"Saved {len(queries)} queries to {queries_path}")

    gt_path = GROUND_TRUTH_DIR / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    logger.info(f"Saved ground truth ({len(ground_truth)} entries) to {gt_path}")

    return {
        "queries_count": len(queries),
        "ground_truth_entries": len(ground_truth),
        "total_relevance_labels": total_relevant,
        "avg_labels_per_query": round(avg_relevant, 1),
        "query_ids": [q["query_id"] for q in queries[:5]],
    }


if __name__ == "__main__":
    result = generate()
    print(json.dumps(result, indent=2))
