#!/usr/bin/env sh
# Verifica/baixa o modelo LLM Ollama configurado no .env
set -e

LLM_MODEL="${OLLAMA_LLM_MODEL:-ibm/granite4:micro}"
EMBEDDINGS_PROVIDER="${EMBEDDINGS_PROVIDER:-local}"

_run() {
  if docker ps --format '{{.Names}}' | grep -q '^rag-ollama$'; then
    docker exec rag-ollama ollama pull "$1"
  else
    ollama pull "$1"
  fi
}

echo "LLM Ollama: $LLM_MODEL"
_run "$LLM_MODEL"

if [ "$EMBEDDINGS_PROVIDER" = "ollama" ]; then
  EMBED_MODEL="${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}"
  echo "Embeddings Ollama: $EMBED_MODEL"
  _run "$EMBED_MODEL"
else
  echo "Embeddings: local ($EMBEDDING_MODEL — sentence-transformers, sem pull Ollama)"
fi

echo "Pronto."
