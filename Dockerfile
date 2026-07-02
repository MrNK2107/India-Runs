FROM python:3.11-slim

WORKDIR /app

# System deps for faiss, psycopg2, spacy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached layer)
COPY pyproject.toml .
RUN mkdir -p src && pip install --no-cache-dir ".[dev]" && rm -rf src

# Copy source code and complete the package installation
COPY . .
RUN pip install --no-cache-dir -e ".[dev]" --no-deps

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Build FAISS + BM25 indexes from the 500-profiles sample
# (the full 100k candidates.jsonl is too large for the container)
RUN python scripts/build_indexes.py \
    --profiles data/samples/sample_500.jsonl \
    --force

# Point ProfileStore to the sample data (full 100k JSONL is too large for the container)
ENV PROFILES_PATH=data/samples/sample_500.jsonl

EXPOSE 7860

# By default serve the API. Override CMD to run the ranker:
#   docker run ... --entrypoint python rank.py --batch --out submission.csv
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]
