from __future__ import annotations

import csv
import gzip
import json
import logging
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProfileParser:
    def __init__(self, normalizer: Any | None = None):
        self.normalizer = normalizer
        self.failed_profiles = 0

    def parse_json(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def parse_jsonl_file(
        self, path: Path, skip_noisy: bool = False, quality_scorer: Any | None = None
    ) -> Generator[dict[str, Any], None, None]:
        open_func = gzip.open if path.suffix == ".gz" else open
        mode = "rt" if path.suffix == ".gz" else "r"
        with open_func(path, mode, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    profile = json.loads(line)
                    if skip_noisy and quality_scorer and self.normalizer:
                        normalized = self.normalizer(profile)
                        score = quality_scorer(normalized)
                        if score < 0.3:
                            self.failed_profiles += 1
                            continue
                    yield profile
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                    self.failed_profiles += 1

    def parse_json_file(self, path: Path) -> list[dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    def parse_docx(self, path: Path) -> dict[str, Any]:
        p = Path(path)
        text_parts: list[str] = []
        with zipfile.ZipFile(path) as z:
            xml_content = z.read("word/document.xml")
            root = ET.fromstring(xml_content)
            for para in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
                texts = [
                    t.text
                    for t in para.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                    if t.text
                ]
                if texts:
                    text_parts.append("".join(texts))
        return {
            "candidate_id": p.stem,
            "profile": {"anonymized_name": p.stem},
            "raw_text": "\n".join(text_parts),
        }

    def parse_csv_file(self, path: Path) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "rank" in row:
                    row["rank"] = int(row["rank"])
                if "score" in row:
                    row["score"] = float(row["score"])
                results.append(row)
        return results

    def parse_csv_stream(self, path: Path) -> Generator[dict[str, Any], None, None]:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "rank" in row:
                    row["rank"] = int(row["rank"])
                if "score" in row:
                    row["score"] = float(row["score"])
                yield row

    def parse_batch(
        self, data: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        successful: list[dict[str, Any]] = []
        errors: list[str] = []
        for i, item in enumerate(data):
            try:
                if not isinstance(item, dict):
                    raise TypeError(f"Expected dict, got {type(item).__name__}")
                successful.append(self.parse_json(item))
            except Exception as e:
                cid = item.get("candidate_id", f"row_{i}") if isinstance(item, dict) else f"row_{i}"
                errors.append(f"{cid}: {e}")
        return successful, errors
