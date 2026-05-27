"""
Filtro leve de contexto pós-rerank (descarta trechos com score muito baixo).
"""
from typing import List

from langchain_core.documents import Document

from src.config import settings


def grade_documents(documents: List[Document]) -> List[Document]:
    """Mantém trechos acima do limiar de rerank (se disponível)."""
    if not documents or not settings.context_grader_enabled:
        return documents

    threshold = settings.context_grader_min_score
    kept = [
        doc
        for doc in documents
        if doc.metadata.get("rerank_score", threshold) >= threshold
    ]
    return kept if kept else documents[: max(1, settings.rerank_top_k // 2)]
