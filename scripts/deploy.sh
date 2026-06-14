#!/usr/bin/env bash
set -euo pipefail

echo "=== India Runs — Deployment Script ==="

# 1. Build Docker image
docker compose build

# 2. Start infrastructure
docker compose up -d postgres redis
sleep 5  # Wait for postgres

# 3. Generate synthetic data (if not exists)
if [ ! -f data/profiles/profiles.json ]; then
    echo "Generating synthetic profiles..."
    docker compose run --rm app python scripts/generate_data.py
fi

# 4. Build indexes
echo "Building FAISS + BM25 indexes..."
docker compose run --rm app python scripts/build_indexes.py

# 5. Run evaluation
echo "Running evaluation..."
docker compose run --rm app python scripts/evaluate.py

# 6. Start application
echo "Starting application..."
docker compose up -d app

echo "=== Application running at http://localhost:8000 ==="
echo "=== Gradio UI at http://localhost:7860 ==="
