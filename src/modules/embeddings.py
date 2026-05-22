"""
Embeddings module for RAG Agent
Uses local sentence-transformers (semantic search in Portuguese).
"""
from typing import List

from langchain_core.embeddings import Embeddings

from src.logger import logger
from src.config import settings, DEFAULT_EMBEDDING_MODEL


class LocalEmbeddings(Embeddings):
    """Multilingual sentence-transformers embeddings (works offline)."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or getattr(
            settings, "embedding_model", DEFAULT_EMBEDDING_MODEL
        )
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        vectors = model.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        model = self._get_model()
        vector = model.encode([text], show_progress_bar=False)
        return vector[0].tolist()


class EmbeddingsFactory:
    """Factory for creating embeddings instances"""

    @staticmethod
    def create_embeddings(provider: str = "local") -> Embeddings:
        if provider in ("local", "deepseek", "huggingface"):
            logger.info("Using local sentence-transformers embeddings")
            return LocalEmbeddings()
        logger.warning(f"Unknown provider: {provider}, using local embeddings")
        return LocalEmbeddings()
