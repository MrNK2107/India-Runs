"""Quick sanity check of a submission CSV (default: ./submission.csv).

Usage: python scripts/diagnostics/quick_validate_csv.py [path/to/submission.csv]
"""
import csv
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
with open(path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f'Total rows: {len(rows)}')
ids = [r['candidate_id'] for r in rows]
print(f'Unique IDs: {len(set(ids))}')
print(f'IDs valid CAND_: {all(i.startswith("CAND_") for i in ids)}')
scores = [float(r['score']) for r in rows]
print(f'Score range: {min(scores):.4f} - {max(scores):.4f}')
print(f'Non-decreasing: {all(scores[i] >= scores[i+1] for i in range(len(scores)-1))}')
ranks = [int(r['rank']) for r in rows]
print(f'Ranks 1-100: {ranks == list(range(1,101))}')
print(f'All have reasoning: {all(len(r["reasoning"]) > 10 for r in rows)}')
