from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any


class ProfileParser:
    def __init__(self, normalizer: Any | None = None):
        self.normalizer = normalizer

    def parse_json(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def parse_jsonl_file(self, path: Path) -> list[dict[str, Any]]:
        open_func = gzip.open if path.suffix == ".gz" else open
        mode = "rt" if path.suffix == ".gz" else "r"
        profiles: list[dict[str, Any]] = []
        with open_func(path, mode, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    profiles.append(json.loads(line))
        return profiles

    def parse_json_file(self, path: Path) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    def parse_batch(
        self, data: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        successful: list[dict[str, Any]] = []
        errors: list[str] = []
        for i, item in enumerate(data):
            try:
                successful.append(self.parse_json(item))
            except Exception as e:
                cid = item.get("candidate_id", f"row_{i}")
                errors.append(f"{cid}: {e}")
        return successful, errors
