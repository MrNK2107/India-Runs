from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from src.core.constants import PROFILES_PATH
from src.core.models import Profile
from src.language.multilingual import MultilingualEmbedder
from src.search.bm25_search import BM25Search
from src.search.vector_search import VectorSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_indexes(profiles_path: Path = PROFILES_PATH) -> None:
    start = time.perf_counter()

    profiles = load_profiles(profiles_path)
    if not profiles:
        logger.error(f"No profiles found at {profiles_path}")
        return
    logger.info(f"Loaded {len(profiles)} profiles")

    raw_texts = [p.raw_text for p in profiles]
    profile_ids = [p.profile_id for p in profiles]
    document_texts = [_build_document_text(p) for p in profiles]

    logger.info("Generating embeddings...")
    embedder = MultilingualEmbedder()
    embeddings = embedder.embed_batch(raw_texts)
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


def load_profiles(path: Path) -> list[Profile]:
    if not path.exists():
        logger.warning(f"Profiles file not found: {path}")
        return []
    with open(path) as f:
        if path.suffix == ".jsonl":
            profiles = [Profile(**json.loads(line)) for line in f if line.strip()]
        else:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            profiles = [Profile(**p) for p in data]
    return profiles


if __name__ == "__main__":
    build_indexes()
