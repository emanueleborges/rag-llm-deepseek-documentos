"""
Vector Store module for RAG Agent
ChromaDB + BM25 híbrido + reranker BGE (Fase 2)
"""
import hashlib
import json
from typing import Any, List, Optional
from pathlib import Path

import chromadb
from langchain_core.documents import Document

from src.logger import logger
from src.config import settings
from src.modules.embeddings import EmbeddingsFactory
from src.modules.retrieval.bm25_index import BM25Index
from src.modules.retrieval.hybrid_retriever import HybridRetriever
from src.modules.retrieval.reranker import BGEReranker
from src.modules.retrieval.context_grader import grade_documents


def _chunk_id(doc: Document, index: int) -> str:
    """Stable unique id per chunk to avoid collisions on re-ingest."""
    explicit = doc.metadata.get("chunk_id")
    if explicit:
        page = doc.metadata.get("page", "")
        digest = hashlib.md5(
            doc.page_content.encode("utf-8", errors="ignore")
        ).hexdigest()[:8]
        return f"{explicit}_p{page}_{digest}"

    source = str(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page", "")
    digest = hashlib.md5(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{Path(source).stem}_p{page}_{index}_{digest}"


def _sanitize_metadata(metadata: dict) -> dict:
    """Chroma aceita apenas str, int, float ou bool."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


class VectorStoreManager:
    """Manage vector store operations with hybrid retrieval."""

    def __init__(
        self,
        embeddings_provider: str = "local",
        collection_name: str = None,
        persist_directory: str = None,
    ):
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or str(settings.vector_store_dir)
        self.embeddings = EmbeddingsFactory.create_embeddings(settings.embeddings_provider)

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        bm25_path = Path(self.persist_directory) / "bm25_index.pkl"
        self.bm25_index = BM25Index(bm25_path)
        self.bm25_index.load()

        self.hybrid_retriever = HybridRetriever(
            vector_search=self.similarity_search,
            bm25_index=self.bm25_index,
        )
        self.reranker = BGEReranker() if settings.reranker_enabled else None

        self._client: chromadb.ClientAPI | None = None
        self._collection = None

        logger.info(
            "Vector store initialized at %s (hybrid=%s, bm25_ready=%s, reranker=%s)",
            self.persist_directory,
            settings.hybrid_search_enabled,
            self.bm25_index.is_ready,
            settings.reranker_enabled,
        )

    def _get_client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    def _get_collection(self):
        if self._collection is None:
            self._collection = self._get_client().get_or_create_collection(
                name=self.collection_name
            )
        return self._collection

    @staticmethod
    def parse_metadata_filter() -> Optional[dict]:
        raw = (settings.retrieval_metadata_filter or "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            logger.warning("RETRIEVAL_METADATA_FILTER inválido (JSON): %s", raw)
            return None

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            logger.warning("No documents to add to vector store")
            return

        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            collection = self._get_collection()
            ids = [_chunk_id(doc, i) for i, doc in enumerate(documents)]
            texts = [doc.page_content for doc in documents]
            metadatas = [_sanitize_metadata(doc.metadata) for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)

            batch_size = 500
            for start in range(0, len(documents), batch_size):
                end = start + batch_size
                collection.add(
                    ids=ids[start:end],
                    documents=texts[start:end],
                    metadatas=metadatas[start:end],
                    embeddings=embeddings[start:end],
                )

            self.bm25_index.build(documents)
            logger.info(f"Successfully added {len(documents)} documents to vector store")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def replace_documents(self, documents: List[Document]) -> None:
        if not documents:
            logger.warning("No documents to replace in vector store")
            return

        try:
            logger.info(f"Replacing collection with {len(documents)} document chunks")
            self.delete_collection()
            self.add_documents(documents)
            logger.info("Collection replaced successfully")

        except Exception as e:
            logger.error(f"Error replacing documents in vector store: {e}")
            raise

    def get_retriever(self, search_kwargs: dict = None):
        if search_kwargs is None:
            search_kwargs = {"k": settings.retrieval_k}

        k = search_kwargs.get("k", settings.retrieval_k)
        manager = self

        class SimpleRetriever:
            def invoke(self, query: str) -> List[Document]:
                return manager.similarity_search(query, k=k)

        return SimpleRetriever()

    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: dict = None,
    ) -> List[Document]:
        k = k or settings.retrieval_k
        query_embedding = self.embeddings.embed_query(query)
        collection = self._get_collection()

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas"],
        }
        if filter:
            kwargs["where"] = filter

        results = collection.query(**kwargs)
        documents: List[Document] = []

        if not results.get("ids") or not results["ids"][0]:
            return documents

        for i, _doc_id in enumerate(results["ids"][0]):
            documents.append(
                Document(
                    page_content=results["documents"][0][i] or "",
                    metadata=results["metadatas"][0][i] or {},
                )
            )
        return documents

    def advanced_search(
        self,
        query: str,
        k: int = None,
        metadata_filter: dict = None,
        rerank_top_k: int = None,
    ) -> List[Document]:
        """
        Busca híbrida (vetor + BM25) → rerank BGE → context grader.
        """
        k = k or settings.hybrid_fetch_k
        meta_filter = metadata_filter if metadata_filter is not None else self.parse_metadata_filter()
        rerank_top_k = rerank_top_k or settings.rerank_top_k

        if settings.hybrid_search_enabled and self.bm25_index.is_ready:
            candidates = self.hybrid_retriever.search(
                query, k=k, metadata_filter=meta_filter
            )
        else:
            candidates = self.similarity_search(
                query, k=max(k, settings.retrieval_k), filter=meta_filter
            )

        if settings.reranker_enabled and self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=rerank_top_k)

        return grade_documents(candidates)

    def get_metadata_sample(self, limit: int = 3) -> List[dict]:
        try:
            result = self._get_collection().peek(limit=limit)
            metas = result.get("metadatas") or []
            return metas[:limit]
        except Exception as exc:
            logger.warning("Não foi possível obter amostra de metadados: %s", exc)
            return []

    def delete_collection(self) -> None:
        try:
            logger.warning(f"Deleting collection: {self.collection_name}")
            try:
                self._get_client().delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = None
            self.bm25_index.clear()
            logger.info("Collection deleted successfully")

        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def get_collection_info(self) -> dict:
        try:
            collection = self._get_collection()
            info = {
                "name": self.collection_name,
                "count": collection.count(),
                "persist_directory": self.persist_directory,
                "backend": settings.vector_store_backend,
                "hybrid_search": settings.hybrid_search_enabled,
                "bm25_ready": self.bm25_index.is_ready,
                "bm25_chunks": len(self.bm25_index.documents),
                "reranker": settings.reranker_enabled,
                "reranker_model": settings.reranker_model if settings.reranker_enabled else None,
                "metadata_fields": sorted(
                    {
                        key
                        for meta in self.get_metadata_sample(limit=5)
                        for key in meta.keys()
                    }
                ),
            }
            logger.info(f"Collection info: {info}")
            return info

        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
