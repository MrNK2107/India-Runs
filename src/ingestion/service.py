"""Runtime profile ingestion: validate, persist, embed and index profiles.

Used by ``POST /api/v1/ingest``. Ingested profiles are:

1. validated and normalised (same ``normalize_redrob`` path as the offline
   index build, so they rank the same way as the bulk data),
2. persisted to a JSONL store (``data/profiles/ingested.jsonl`` by default),
   one record per profile id, written atomically,
3. embedded and upserted into the live FAISS and BM25 indexes and the
   ``ProfileStore``, so the search/rerank path can retrieve them at once.

Ingest is idempotent by profile id: re-sending an identical profile is a
no-op (``unchanged``); a changed profile replaces the old one in the store
and both indexes (``updated``). On startup ``load()`` replays the JSONL store
into the freshly loaded indexes, so ingested profiles survive restarts
without rewriting the (large) prebuilt index files.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.core.models import Profile
from src.ingestion.normalizer import normalize_redrob

logger = logging.getLogger(__name__)

MAX_PROFILES_PER_REQUEST = 1000
_ID_KEYS = ("candidate_id", "profile_id", "id")
_MAX_ID_LEN = 128


class _Embedder(Protocol):
    def embed_batch(self, texts: list[str], batch_size: int = ...) -> np.ndarray[Any, Any]: ...


class _VectorIndex(Protocol):
    def upsert(self, embeddings: np.ndarray[Any, Any], profile_ids: list[str]) -> None: ...


class _TextIndex(Protocol):
    def upsert(self, documents: list[str], profile_ids: list[str]) -> None: ...


class _Store(Protocol):
    def add_ingested(self, profiles: list[Profile]) -> None: ...


def build_document_text(profile: Profile) -> str:
    """Text indexed by BM25 for a profile (shared with scripts/build_indexes.py)."""
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


def content_hash(raw: dict[str, Any]) -> str:
    canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ProfileValidationError(ValueError):
    pass


def extract_profile_id(raw: dict[str, Any]) -> str:
    """Stable id for a raw profile; required so re-ingest is idempotent.

    (``normalize_redrob`` falls back to a random UUID, which would create a
    duplicate on every re-ingest, so a missing id is a validation error.)
    """
    for key in _ID_KEYS:
        val = raw.get(key)
        if val is None or isinstance(val, bool) or isinstance(val, (dict, list)):
            continue
        pid = str(val).strip()
        if pid:
            if len(pid) > _MAX_ID_LEN:
                raise ProfileValidationError(f"'{key}' is longer than {_MAX_ID_LEN} chars")
            return pid
    raise ProfileValidationError(
        "missing profile id (one of 'candidate_id', 'profile_id', 'id')"
    )


def validate_and_normalize(raw: Any) -> tuple[str, Profile]:
    if not isinstance(raw, dict):
        raise ProfileValidationError(f"expected a JSON object, got {type(raw).__name__}")
    pid = extract_profile_id(raw)
    for key, kind in (("profile", dict), ("skills", list), ("career_history", list),
                      ("education", list), ("certifications", list), ("languages", list),
                      ("redrob_signals", dict)):
        if key in raw and raw[key] is not None and not isinstance(raw[key], kind):
            raise ProfileValidationError(f"'{key}' must be a {kind.__name__}")
    for key, required in (("skills", "name"), ("career_history", "title"),
                          ("career_history", "company"), ("education", "institution")):
        for j, entry in enumerate(raw.get(key) or []):
            if not isinstance(entry, dict) or not entry.get(required):
                raise ProfileValidationError(f"'{key}[{j}]' needs a non-empty '{required}'")
    # normalize_redrob reads the id from candidate_id only.
    normalized_input = {**raw, "candidate_id": pid}
    for key in ("skills", "career_history", "education", "certifications", "languages"):
        if normalized_input.get(key) is None:
            normalized_input.pop(key, None)
    if normalized_input.get("profile") is None:
        normalized_input["profile"] = {}
    try:
        profile = normalize_redrob(normalized_input)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise ProfileValidationError(f"could not normalise profile: {e}") from e
    prof = normalized_input["profile"]
    has_text = any(
        isinstance(prof.get(k), str) and prof[k].strip()
        for k in ("summary", "headline", "current_title")
    )
    if not has_text and not profile.skills and not profile.experience:
        raise ProfileValidationError("profile has no searchable content")
    return pid, profile


@dataclass
class IngestResult:
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def successful(self) -> int:
        return len(self.created) + len(self.updated) + len(self.unchanged)


class IngestService:
    def __init__(
        self,
        profile_store: _Store,
        vector_search: _VectorIndex,
        bm25_search: _TextIndex,
        embedder: _Embedder,
        store_path: Path,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.profile_store = profile_store
        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self.embedder = embedder
        self.store_path = store_path
        self._on_change = on_change
        self._lock = threading.Lock()
        # profile id -> {"hash": ..., "raw": ...}
        self._records: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._records)

    # ── persistence ─────────────────────────────────────────────────

    def load(self) -> int:
        """Replay the persisted store into the indexes. Returns profiles loaded."""
        with self._lock:
            records = self._read_store()
            profiles: list[Profile] = []
            for pid, rec in records.items():
                try:
                    _, profile = validate_and_normalize(rec["raw"])
                except ProfileValidationError as e:
                    logger.warning("Skipping stored profile %s: %s", pid, e)
                    continue
                profiles.append(profile)
                self._records[pid] = rec
            self._index(profiles)
        logger.info("Loaded %d ingested profiles from %s", len(profiles), self.store_path)
        return len(profiles)

    def _read_store(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        if not self.store_path.exists():
            return records
        with open(self.store_path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    pid = str(rec["profile_id"])
                    raw = rec["raw"]
                    if not isinstance(raw, dict):
                        raise TypeError("raw is not an object")
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning("Bad line %d in %s: %s", lineno, self.store_path, e)
                    continue
                records[pid] = {"hash": rec.get("hash") or content_hash(raw), "raw": raw}
        return records

    def _write_store(self, records: dict[str, dict[str, Any]]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for pid, rec in records.items():
                line = {"profile_id": pid, "hash": rec["hash"], "raw": rec["raw"]}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        os.replace(tmp, self.store_path)

    # ── ingest ──────────────────────────────────────────────────────

    def ingest(self, items: list[Any]) -> IngestResult:
        result = IngestResult()
        batch: dict[str, tuple[str, dict[str, Any], Profile]] = {}
        for i, raw in enumerate(items):
            try:
                pid, profile = validate_and_normalize(raw)
            except ProfileValidationError as e:
                result.errors.append(f"Item {i}: {e}")
                continue
            # A later duplicate id in the same upload wins.
            batch[pid] = (content_hash(raw), raw, profile)

        with self._lock:
            to_index: list[Profile] = []
            new_records = dict(self._records)
            for pid, (digest, raw, profile) in batch.items():
                existing = self._records.get(pid)
                if existing is not None and existing["hash"] == digest:
                    result.unchanged.append(pid)
                    continue
                (result.updated if existing is not None else result.created).append(pid)
                new_records[pid] = {"hash": digest, "raw": raw}
                to_index.append(profile)
            if not to_index:
                return result
            # Persist first: if indexing fails the next startup replays the
            # store, so the profile isn't lost.
            self._write_store(new_records)
            self._records = new_records
            self._index(to_index)
        if self._on_change is not None:
            self._on_change()
        return result

    def _index(self, profiles: list[Profile]) -> None:
        if not profiles:
            return
        pids = [p.profile_id for p in profiles]
        embeddings = self.embedder.embed_batch([p.raw_text or "" for p in profiles])
        self.vector_search.upsert(np.asarray(embeddings, dtype=np.float32), pids)
        self.bm25_search.upsert([build_document_text(p) for p in profiles], pids)
        self.profile_store.add_ingested(profiles)
