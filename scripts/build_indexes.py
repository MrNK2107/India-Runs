from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import DATA_DIR
from src.core.constants import CANDIDATES_PATH, SAMPLE_PATH
from src.core.models import Profile
from src.ingestion.normalizer import normalize_redrob
from src.ingestion.parser import ProfileParser
from src.ingestion.quality_scorer import compute_data_quality_score
from src.ingestion.service import build_document_text
from src.language.multilingual import create_embedder
from src.search.bm25_search import BM25Search
from src.search.vector_search import VectorSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_indexes(
    profiles_path: Path = CANDIDATES_PATH,
    sample_count: int = 0,
    force: bool = False,
    embed_batch_size: int = 64,
) -> None:
    start = time.perf_counter()

    if not profiles_path.exists():
        logger.warning(f"Profiles file not found: {profiles_path}")
        logger.info(f"Falling back to sample data: {SAMPLE_PATH}")
        profiles_path = SAMPLE_PATH

    if not profiles_path.exists():
        logger.error(f"No data found at {profiles_path} or {CANDIDATES_PATH}")
        return

    index_dir = DATA_DIR / "indexes"
    faiss_path = index_dir / "faiss_index.bin"
    bm25_path = index_dir / "bm25_index.pkl"

    if faiss_path.exists() and bm25_path.exists() and not force:
        logger.info("Indexes already exist. Use --force to rebuild.")
        return

    logger.info(f"Loading profiles from {profiles_path}")
    parser = ProfileParser()
    profiles: list[Profile] = []
    loaded = 0
    skipped = 0

    if profiles_path.suffix == ".jsonl":
        log_interval = max(1, sample_count // 10) if sample_count > 0 else 10000
        for raw in parser.parse_jsonl_file(profiles_path):
            try:
                normalized = normalize_redrob(raw)
                qs = compute_data_quality_score(normalized)
                if qs < 0.3:
                    skipped += 1
                    continue
                profiles.append(normalized)
                loaded += 1
                if loaded % log_interval == 0 and sample_count == 0:
                    logger.info(f"  Loaded {loaded} profiles ({skipped} skipped so far)...")
            except Exception:
                skipped += 1
                continue
            if sample_count > 0 and loaded >= sample_count:
                break
    else:
        data = parser.parse_json_file(profiles_path)
        for item in data:
            try:
                normalized = normalize_redrob(item)
                profiles.append(normalized)
                loaded += 1
            except Exception:
                skipped += 1
            if sample_count > 0 and loaded >= sample_count:
                break
    logger.info(f"Loaded {len(profiles)} profiles ({skipped} skipped)")

    if not profiles:
        logger.error("No valid profiles to index")
        return

    raw_texts = [p.raw_text for p in profiles]
    profile_ids = [p.profile_id for p in profiles]
    document_texts = [build_document_text(p) for p in profiles]

    logger.info("Generating embeddings (this may take a while)...")
    embedder = create_embedder()
    batch_size = 500
    all_embeddings = []
    num_batches = (len(raw_texts) + batch_size - 1) // batch_size
    with tqdm(total=num_batches, desc="Embedding", unit="batch") as pbar:
        for i in range(0, len(raw_texts), batch_size):
            batch = raw_texts[i : i + batch_size]
            batch_emb = embedder.embed_batch(batch, batch_size=embed_batch_size)
            all_embeddings.append(batch_emb)
            pbar.update(1)
    import numpy as np
    embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
    logger.info(f"Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

    vector_search = VectorSearch(dimension=384)
    vector_search.build_index(embeddings, profile_ids)
    vector_search.save()
    logger.info(f"FAISS index saved: {vector_search.size} vectors")

    bm25_search = BM25Search()
    bm25_search.build_index(document_texts, profile_ids)
    bm25_search.save()
    logger.info(f"BM25 index saved: {bm25_search.size} documents")

    offset_path = index_dir / "offset_index.json"
    _save_offset_index(profiles_path, profile_ids, offset_path)

    elapsed = time.perf_counter() - start
    logger.info(f"All indexes built successfully in {elapsed:.1f}s")


def _save_offset_index(
    profiles_path: Path, profile_ids: list[str], output_path: Path,
) -> None:
    import json
    if profiles_path.suffix != ".jsonl":
        logger.info("Skipping offset index (not a JSONL file)")
        return
    pid_set = set(profile_ids)
    offsets: dict[str, int] = {}
    with open(profiles_path, encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                cand_id = (
                    raw.get("profile_id")
                    or raw.get("candidate_id")
                    or raw.get("id")
                )
                profile_nested = raw.get("profile", {})
                if isinstance(profile_nested, dict) and not cand_id:
                    cand_id = (
                        profile_nested.get("profile_id")
                        or profile_nested.get("candidate_id")
                        or profile_nested.get("id")
                    )
                if cand_id and str(cand_id) in pid_set:
                    offsets[str(cand_id)] = offset
            except json.JSONDecodeError:
                continue
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(offsets, f)
    logger.info(f"Offset index saved: {len(offsets)} entries")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indexes from profiles")
    parser.add_argument("--profiles", type=str, default=None,
                        help="Path to candidate profiles file (JSON or JSONL)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Process only N profiles (for quick testing)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild indexes even if they exist")
    parser.add_argument("--embed-batch-size", type=int, default=64,
                        help="Texts per model forward pass (lower = less memory)")
    args = parser.parse_args()

    profiles_p = Path(args.profiles) if args.profiles else CANDIDATES_PATH
    build_indexes(profiles_path=profiles_p, sample_count=args.sample, force=args.force,
                  embed_batch_size=args.embed_batch_size)


if __name__ == "__main__":
    main()
