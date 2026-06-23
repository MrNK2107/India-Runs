"""Build offset index only — does not rebuild FAISS/BM25."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import DATA_DIR

profiles_path = DATA_DIR / "profiles" / "candidates.jsonl"
output_path = DATA_DIR / "indexes" / "offset_index.json"

id_map_path = DATA_DIR / "indexes" / "faiss_id_map.json"
if not id_map_path.exists():
    print("FAISS id map not found. Run build_indexes.py first.")
    sys.exit(1)

with open(id_map_path) as f:
    profile_ids = json.load(f)

pid_set = set(profile_ids)
print(f"Looking up offsets for {len(pid_set)} profile IDs...")

start = time.perf_counter()
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

elapsed = time.perf_counter() - start
print(f"Found {len(offsets)}/{len(pid_set)} profile offsets in {elapsed:.1f}s")
missing = pid_set - set(offsets.keys())
if missing:
    print(f"Missing {len(missing)} profiles")

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(offsets, f)
print(f"Offset index saved to {output_path}")
