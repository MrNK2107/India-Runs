# Kasauti (india-runs) - evidence-backed candidate ranking.
#
#   docker build -t kasauti .                      # lite engine (default): small, CPU, no model download
#   docker build -t kasauti --build-arg ENGINE=full .   # transformer embeddings + cross-encoder + LLM agents
#   docker run -p 7860:7860 kasauti                # open http://localhost:7860
#
# With no prebuilt index the app indexes data/samples/sample_500.jsonl in memory at
# startup (seconds), so the container is useful immediately. Mount a volume on
# /app/data/profiles to keep profiles added through POST /api/v1/ingest.
FROM python:3.11-slim

ARG ENGINE=lite
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENBLAS_NUM_THREADS=2 \
    PORT=7860

WORKDIR /app

COPY pyproject.toml requirements-lite.txt ./
RUN if [ "$ENGINE" = "full" ]; then \
        apt-get update && apt-get install -y --no-install-recommends gcc build-essential libpq-dev \
        && rm -rf /var/lib/apt/lists/* \
        && mkdir -p src && pip install "." && rm -rf src; \
    else \
        pip install -r requirements-lite.txt; \
    fi

COPY . .
RUN pip install --no-deps -e . \
    && useradd --create-home --uid 1000 app \
    && mkdir -p data/profiles data/indexes && chown -R app:app data

# Engine defaults live in /app/.engine.env and only fill variables that are unset, so
# `docker run -e EMBEDDING_BACKEND=...` and compose .env values always win.
ENV AUTO_INDEX=true \
    HF_HOME=/app/.hf
RUN if [ "$ENGINE" = "full" ]; then b=transformer; ce=true; else b=hashing; ce=false; fi \
    && printf ': "${EMBEDDING_BACKEND:=%s}"\n: "${CROSS_ENCODER_ENABLED:=%s}"\nexport EMBEDDING_BACKEND CROSS_ENCODER_ENABLED\n' "$b" "$ce" > /app/.engine.env
# The full engine runs with HF_HUB_OFFLINE=1 (src/main.py), so bake the models in at build time.
RUN if [ "$ENGINE" = "full" ]; then \
        python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
        && chown -R app:app /app/.hf; \
    fi

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/api/v1/health',timeout=4)"

CMD ["sh", "-c", ". /app/.engine.env; exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
