from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, TextIO

from src.core.config import DATA_DIR
from src.core.constants import SAMPLE_PATH
from src.core.models import Profile
from src.ingestion.normalizer import normalize_redrob

logger = logging.getLogger(__name__)


class ProfileStore:
    def __init__(self, jsonl_path: Path | None = None, max_cache: int = 500) -> None:
        self._max_cache = max_cache
        self._offset_index: dict[str, int] = {}
        self._cache: OrderedDict[str, Profile] = OrderedDict()
        self._index_built = False
        self._sample_profiles: dict[str, Profile] = {}
        # Profiles added at runtime via POST /ingest; they take precedence
        # over the JSONL file and samples (a re-ingest replaces a profile).
        self._ingested: dict[str, Profile] = {}
        self._file_handle: TextIO | None = None
        # Auto-detect best available data source
        # Respect PROFILES_PATH env var (set by Dockerfile for HF Spaces)
        env_path = os.environ.get("PROFILES_PATH")
        if env_path:
            detected = Path(env_path)
        else:
            candidates = DATA_DIR / "profiles" / "candidates.jsonl"
            sample_500 = DATA_DIR / "samples" / "sample_500.jsonl"
            sample_100 = DATA_DIR / "samples" / "sample_candidates.json"
            detected = (
                candidates if candidates.exists()
                else sample_500 if sample_500.exists()
                else sample_100
            )
        self.path = jsonl_path or detected
        self._auto_init_samples()
        self._auto_load_offset_index()

    def _auto_load_offset_index(self) -> None:
        offset_path = DATA_DIR / "indexes" / "offset_index.json"
        if offset_path.exists() and not self._index_built:
            self.load_offset_index(offset_path)

    def _auto_init_samples(self) -> None:
        """Automatically load sample profiles alongside the main profile file.

        Samples are always loaded so they are available for quick access,
        even when the main 100K profile JSONL is present.
        """
        sample_path = SAMPLE_PATH
        if not sample_path.exists():
            logger.warning(f"No sample data found at {sample_path}")
            return
        self.load_sample(sample_path)

    def load_sample(self, sample_path: Path) -> None:
        if not sample_path.exists():
            return
        with open(sample_path, encoding="utf-8") as f:
            data = json.load(f)
            profiles_list = data if isinstance(data, list) else [data]
            for p in profiles_list:
                try:
                    profile = normalize_redrob(p)
                    pid = profile.profile_id
                    self._sample_profiles[pid] = profile
                    if pid not in self._offset_index:
                        self._offset_index[pid] = -1
                except Exception:
                    pass
        logger.info(f"Loaded {len(self._sample_profiles)} sample profiles")

    def load_offset_index(self, index_path: Path) -> None:
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                loaded: dict[str, int] = json.load(f)
            # Keep sample IDs registered by load_sample() (offset -1) that the
            # saved index doesn't know about, instead of silently dropping them.
            for pid, offset in self._offset_index.items():
                loaded.setdefault(pid, offset)
            self._offset_index = loaded
            self._index_built = True
            logger.info(f"Loaded offset index with {len(self._offset_index)} entries")

    def save_offset_index(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._offset_index, f)
        logger.info(f"Saved offset index ({len(self._offset_index)} entries) to {path}")

    def _build_offset_index(self) -> None:
        if self._index_built:
            return
        if not self.path.exists():
            logger.warning(f"Profiles file not found: {self.path}")
            self._index_built = True
            return
        count = 0
        with open(self.path, encoding="utf-8") as f:
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
                    if not isinstance(raw, dict):
                        continue
                    pid = self._extract_id(raw)
                    if pid and pid not in self._offset_index:
                        self._offset_index[pid] = offset
                        count += 1
                except json.JSONDecodeError:
                    pass
        self._index_built = True
        logger.info(f"Built offset index: {count} profile IDs")

    @staticmethod
    def _extract_id(raw: dict[str, Any]) -> str | None:
        for key in ("profile_id", "candidate_id", "id"):
            val = raw.get(key)
            if val:
                return str(val)
        profile_nested = raw.get("profile", {})
        if isinstance(profile_nested, dict):
            for key in ("profile_id", "candidate_id", "id"):
                val = profile_nested.get(key)
                if val:
                    return str(val)
        return None

    def add_samples(self, profiles: list[Profile]) -> None:
        """Register already-normalised sample profiles (the in-memory demo engine)."""
        for profile in profiles:
            self._sample_profiles[profile.profile_id] = profile
            self._offset_index.setdefault(profile.profile_id, -1)

    def add_ingested(self, profiles: list[Profile]) -> None:
        """Register (or replace) profiles that arrived through ingestion."""
        for profile in profiles:
            pid = profile.profile_id
            self._ingested[pid] = profile
            self._cache.pop(pid, None)  # drop any stale cached version

    def get_ingested(self) -> dict[str, Profile]:
        return dict(self._ingested)

    def get(self, pid: str) -> Profile | None:
        if pid in self._ingested:
            return self._ingested[pid]
        if pid in self._cache:
            self._cache.move_to_end(pid)
            return self._cache[pid]

        if pid in self._sample_profiles:
            profile = self._sample_profiles[pid]
            if len(self._cache) >= self._max_cache:
                self._cache.popitem(last=False)
            self._cache[pid] = profile
            return profile

        self._build_offset_index()
        offset = self._offset_index.get(pid)
        if offset is None or offset < 0:
            return None

        try:
            if self._file_handle is None or self._file_handle.closed:
                self._file_handle = open(self.path, encoding="utf-8")
            self._file_handle.seek(offset)
            line = self._file_handle.readline()
            raw = json.loads(line)
            profile = normalize_redrob(raw)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

        if len(self._cache) >= self._max_cache:
            self._cache.popitem(last=False)
        self._cache[pid] = profile
        return profile

    def get_all_sample(self) -> dict[str, Profile]:
        return dict(self._sample_profiles)

    def get_all_pids(self) -> list[str]:
        self._build_offset_index()
        pids = list(self._offset_index.keys())
        known = set(pids)
        for pid in [*self._sample_profiles, *self._ingested]:
            if pid not in known:
                known.add(pid)
                pids.append(pid)
        return pids

    def __contains__(self, pid: str) -> bool:
        if pid in self._ingested or pid in self._sample_profiles or pid in self._cache:
            return True
        self._build_offset_index()
        return pid in self._offset_index

    def __len__(self) -> int:
        self._build_offset_index()
        # Sample IDs are also registered in the offset index (offset -1).
        return len(
            self._offset_index.keys() | self._sample_profiles.keys() | self._ingested.keys()
        )

    def __del__(self) -> None:
        if hasattr(self, "_file_handle") and self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass
