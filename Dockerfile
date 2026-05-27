# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --prefix=/install -r requirements-docker.txt \
    && find /install -type d -name __pycache__ -prune -exec rm -rf {} + \
    && find /install -type d -name tests -prune -exec rm -rf {} +

FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PDF_PARSER=pypdf \
    EMBEDDINGS_PROVIDER=ollama \
    RERANKER_ENABLED=false \
    LANGGRAPH_ENABLED=false \
    HYBRID_SEARCH_ENABLED=true

COPY --from=builder /install /usr/local

COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY src/ ./src/

RUN mkdir -p data/documents data/vector_store logs \
    && python -c "import chromadb, langchain_ollama, streamlit; print('deps OK')"

EXPOSE 8501

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
