#!/bin/sh
set -e

# Corrige imagens antigas sem langchain-text-splitters
if ! python -c "import langchain_text_splitters" >/dev/null 2>&1; then
    echo "[entrypoint] Instalando pacotes LangChain ausentes..."
    pip install --no-cache-dir \
        langchain-core==0.2.43 \
        langchain-community==0.2.19 \
        langchain-text-splitters==0.2.4
fi

exec "$@"
