"""Zero-setup engine: build the search indexes in memory from a bundled sample.

Used when no prebuilt FAISS index exists (fresh clone, Docker, CI, demos), so the
product answers its first query seconds after `uvicorn src.main:app` instead of
asking the user to run an index build first. With EMBEDDING_BACKEND=hashing the
whole thing is numpy-only and needs no model download.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.core.constants import SAMPLE_PATH
from src.core.models import Profile
from src.ingestion.normalizer import normalize_redrob
from src.ingestion.parser import ProfileParser
from src.ingestion.service import build_document_text

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE = SAMPLE_PATH.parent / "sample_500.jsonl"


def load_sample_profiles(path: Path, limit: int = 0) -> list[Profile]:
    parser = ProfileParser()
    items: list[Any]
    if path.suffix == ".jsonl":
        items = list(parser.parse_jsonl_file(path))
    else:
        items = list(parser.parse_json_file(path))
    profiles: list[Profile] = []
    seen: set[str] = set()
    for raw in items:
        try:
            profile = normalize_redrob(raw)
        except Exception:  # noqa: BLE001 - a bad sample row must not stop startup
            continue
        if profile.profile_id in seen:
            continue
        seen.add(profile.profile_id)
        profiles.append(profile)
        if limit and len(profiles) >= limit:
            break
    return profiles


def build_sample_engine(
    sample_path: Path = DEFAULT_SAMPLE, limit: int = 0,
) -> tuple[Any, Any, Any, list[Profile]]:
    """Returns (embedder, VectorSearch, BM25Search, profiles)."""
    from src.language.multilingual import create_embedder
    from src.search.bm25_search import BM25Search
    from src.search.vector_search import VectorSearch

    start = time.perf_counter()
    profiles = load_sample_profiles(sample_path, limit)
    if not profiles:
        raise RuntimeError(f"No usable profiles in {sample_path}")
    embedder = create_embedder()
    ids = [p.profile_id for p in profiles]
    vectors = np.asarray(
        embedder.embed_batch([p.raw_text or "" for p in profiles], batch_size=32),
        dtype=np.float32,
    )
    vector_search = VectorSearch(dimension=vectors.shape[1])
    vector_search.build_index(vectors, ids)
    bm25 = BM25Search()
    bm25.build_index([build_document_text(p) for p in profiles], ids)
    logger.info(
        "Built in-memory sample engine: %d profiles from %s in %.1fs",
        len(profiles), sample_path.name, time.perf_counter() - start,
    )
    return embedder, vector_search, bm25, profiles
