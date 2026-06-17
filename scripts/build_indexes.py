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
from src.language.multilingual import MultilingualEmbedder
from src.search.bm25_search import BM25Search
from src.search.vector_search import VectorSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_indexes(
    profiles_path: Path = CANDIDATES_PATH,
    sample_count: int = 0,
    force: bool = False,
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
    logger.info(f"Loaded {len(profiles)} profiles ({skipped} skipped)")

    if not profiles:
        logger.error("No valid profiles to index")
        return

    raw_texts = [p.raw_text for p in profiles]
    profile_ids = [p.profile_id for p in profiles]
    document_texts = [_build_document_text(p) for p in profiles]

    logger.info("Generating embeddings (this may take a while)...")
    embedder = MultilingualEmbedder()
    batch_size = 500
    all_embeddings = []
    num_batches = (len(raw_texts) + batch_size - 1) // batch_size
    with tqdm(total=num_batches, desc="Embedding", unit="batch") as pbar:
        for i in range(0, len(raw_texts), batch_size):
            batch = raw_texts[i : i + batch_size]
            batch_emb = embedder.embed_batch(batch)
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

    elapsed = time.perf_counter() - start
    logger.info(f"All indexes built successfully in {elapsed:.1f}s")


def _build_document_text(profile: Profile) -> str:
    parts: list[str] = []
    if profile.raw_text:
        parts.append(profile.raw_text)
    parts.extend(s.name for s in profile.skills)
    for exp in profile.experience:
        parts.append(exp.title)
        parts.append(exp.company)
        parts.append(exp.description)
    for edu in profile.education:
        parts.append(edu.institution)
        if edu.field:
            parts.append(edu.field)
    if profile.professional and profile.professional.current_title:
        parts.append(profile.professional.current_title)
    if profile.professional and profile.professional.current_company:
        parts.append(profile.professional.current_company)
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indexes from profiles")
    parser.add_argument("--sample", type=int, default=0,
                        help="Process only N profiles (for quick testing)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild indexes even if they exist")
    args = parser.parse_args()

    build_indexes(sample_count=args.sample, force=args.force)


if __name__ == "__main__":
    main()
