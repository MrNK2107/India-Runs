# Deployment Guide — India Runs

## 1. Local Development

### Prerequisites
- Python 3.11+
- Docker & Docker Compose

### Steps

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Start PostgreSQL and Redis
docker compose up -d postgres redis

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys if needed

# 4. Build search indexes
python scripts/build_indexes.py

# 5. Start the API server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 6. In another terminal, start the UI
python src/ui/app.py
```

## 2. Docker Deployment

### Full Stack

```bash
# Build and start all services
docker compose up --build

# This starts:
# - PostgreSQL on port 5432
# - Redis on port 6379
# - FastAPI on port 8000
# - Gradio UI on port 7860
```

### Individual Services

```bash
# API only
docker build -t india-runs-api .
docker run -p 8000:8000 -v $(pwd)/data:/app/data india-runs-api

# With custom config
docker run -p 8000:8000 \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/data:/app/data \
  -e OPENAI_API_KEY=sk-... \
  india-runs-api
```

## 3. HuggingFace Spaces (Gradio UI)

The Gradio UI can be deployed to HuggingFace Spaces for free:

1. Create a Space at https://huggingface.co/new-space
2. Choose Gradio SDK
3. Configure environment:
   - `OPENAI_API_KEY`: Your OpenAI key (optional)
   - `LLM_PROVIDER`: `openai`, `gemini`, or `ollama`
4. The Space will auto-build from the repository

Note: Index building requires running `scripts/build_indexes.py` during Space startup. For large indexes (100K profiles), pre-build locally and commit the index files.

## 4. Railway / Render

### Railway

```bash
# Deploy using the Dockerfile
railway up

# Set environment variables in Railway dashboard:
# - DATABASE_URL
# - REDIS_URL
# - OPENAI_API_KEY
```

### Render

1. Create a new Web Service
2. Connect your GitHub repository
3. Set build command: `pip install -e ".[dev]"`
4. Set start command: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in the dashboard

## 5. Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | No | `postgresql://postgres:postgres@localhost:5432/india_runs` | PostgreSQL connection string |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection string |
| `LLM_PROVIDER` | No | `openai` | LLM provider: `openai`, `gemini`, or `ollama` |
| `OPENAI_API_KEY` | If using OpenAI | — | OpenAI API key |
| `GEMINI_API_KEY` | If using Gemini | — | Google Gemini API key |
| `OLLAMA_BASE_URL` | If using Ollama | `http://localhost:11434` | Ollama server URL |

## System Requirements

- **Memory**: 4 GB minimum (16 GB recommended for 100K profiles)
- **CPU**: 2+ cores
- **Disk**: 2 GB for indexes + models
- **Network**: Required only for LLM API calls (optional)
- **Runtime**: ~3-5 minutes for full index build + evaluation on 100K profiles
