"""Estado compartilhado do grafo RAG."""
from typing import List, TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    question: str
    conversation_context: str
    queries: List[str]
    documents: List[Document]
    mode: str
    relevance_note: str
    answer: str
    retrieval_attempts: int
    next_step: str
    web_context: str
    graph_trace: List[str]
