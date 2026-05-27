"""
Embeddings module for RAG Agent
Suporta sentence-transformers (local) ou Ollama (gratuito).
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


class OllamaEmbeddingsWrapper(Embeddings):
    """OllamaEmbeddings via langchain-ollama."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        from langchain_ollama import OllamaEmbeddings

        self.model = model or settings.ollama_embedding_model
        self.base_url = base_url or settings.ollama_base_url
        self._embeddings = OllamaEmbeddings(
            model=self.model,
            base_url=self.base_url,
        )
        self._query_cache: dict[str, List[float]] = {}
        logger.info("Ollama embeddings: %s @ %s", self.model, self.base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        key = text.strip()
        cached = self._query_cache.get(key)
        if cached is not None:
            return cached
        vec = self._embeddings.embed_query(text)
        if len(self._query_cache) >= 64:
            self._query_cache.clear()
        self._query_cache[key] = vec
        return vec


class FastEmbedWrapper(Embeddings):
    """Embeddings ONNX leves (sem PyTorch) — requer pip install fastembed."""

    _FALLBACK_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding

        requested = model_name or settings.fastembed_model
        supported = {
            m["model"] if isinstance(m, dict) else m
            for m in TextEmbedding.list_supported_models()
        }
        if requested not in supported:
            logger.warning(
                "Modelo FastEmbed '%s' não suportado; usando '%s'.",
                requested,
                self._FALLBACK_MODEL,
            )
            self.model_name = self._FALLBACK_MODEL
        else:
            self.model_name = requested

        self._model = TextEmbedding(model_name=self.model_name)
        logger.info("FastEmbed embeddings: %s", self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(self._model.embed([text]))[0].tolist()


class EmbeddingsFactory:
    """Factory for creating embeddings instances"""

    @staticmethod
    def create_embeddings(provider: str | None = None) -> Embeddings:
        provider = (provider or settings.embeddings_provider or "fastembed").lower()

        if provider == "ollama":
            logger.info("Using Ollama embeddings")
            return OllamaEmbeddingsWrapper()

        if provider == "fastembed":
            logger.info("Using FastEmbed embeddings (leve)")
            return FastEmbedWrapper()

        if provider in ("local", "deepseek", "huggingface", "sentence-transformers"):
            logger.info("Using sentence-transformers embeddings")
            return LocalEmbeddings()

        logger.warning("Unknown embeddings provider '%s', using fastembed", provider)
        return FastEmbedWrapper()
