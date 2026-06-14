# Evaluation — India Runs

## Overview

The evaluation pipeline measures search and matching quality using standard information retrieval metrics. It compares system-ranked results against ground-truth relevance judgments.

## Running Evaluation

```bash
# Basic evaluation
python scripts/evaluate.py

# With custom data
python scripts/evaluate.py --profiles data/profiles.jsonl --queries data/queries.json
```

## Metrics

### 1. Precision@k (P@k)
Fraction of top-k results that are relevant. Measures ranking precision.

```
P@k = |{relevant docs in top-k}| / k
```

### 2. Recall@k (R@k)
Fraction of all relevant documents found in top-k results.

```
R@k = |{relevant docs in top-k}| / |{all relevant docs}|
```

### 3. Mean Reciprocal Rank (MRR)
Average of reciprocal ranks of the first relevant result. Measures how quickly the system finds a good match.

```
MRR = (1/N) * Σ (1 / rank_of_first_relevant)
```

### 4. nDCG@k (Normalized Discounted Cumulative Gain)
Accounts for graded relevance (not just binary). Penalizes relevant results appearing lower in the ranking.

### 5. Cross-lingual MRR
MRR computed specifically for queries and profiles in different languages. Measures multilingual search effectiveness.

### 6. Latency Statistics
p50, p95, p99, and mean latency for the full search pipeline.

## Evaluation Script

`scripts/evaluate.py` loads profiles and ground-truth queries, runs the hybrid search pipeline, and reports:

```
=== Evaluation Results ===
Query: "Senior Python Developer"
  P@5:    0.800
  R@5:    0.571
  MRR:    1.000
  nDCG@5: 0.876
  Latency (ms): p50=45, p95=82, p99=120, mean=52

=== Cross-lingual MRR ===
  English→English: 0.892
  Hindi→English:   0.721

=== Overall (50 queries) ===
  Mean P@5:    0.742
  Mean R@5:    0.614
  Mean MRR:    0.883
  Mean nDCG@5: 0.801
```

## Interpreting Results

| Metric | Good | Target |
|--------|------|--------|
| P@5 | >0.7 | >0.8 |
| R@5 | >0.5 | >0.7 |
| MRR | >0.8 | >0.9 |
| nDCG@5 | >0.7 | >0.85 |
| Cross-lingual MRR | >0.6 | >0.8 |
| p50 latency | <100ms | <50ms |
| p99 latency | <500ms | <200ms |

## Test Suite

Unit and integration tests are located in `tests/` and run via pytest:

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src

# Specific module
pytest tests/test_search/ -v
```

All 57 tests pass as of the latest run.
