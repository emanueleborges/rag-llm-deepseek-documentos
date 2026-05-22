"""
Vector Store module for RAG Agent
Manages ChromaDB vector store
"""
import hashlib
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.logger import logger
from src.config import settings
from src.modules.embeddings import EmbeddingsFactory


def _chunk_id(doc: Document, index: int) -> str:
    """Stable unique id per chunk to avoid collisions on re-ingest."""
    source = str(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page", "")
    digest = hashlib.md5(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{Path(source).stem}_p{page}_{index}_{digest}"


class VectorStoreManager:
    """Manage vector store operations"""

    def __init__(
        self,
        embeddings_provider: str = "local",
        collection_name: str = None,
        persist_directory: str = None,
    ):
        """
        Initialize vector store manager

        Args:
            embeddings_provider: Embeddings provider name
            collection_name: ChromaDB collection name
            persist_directory: Directory to persist vector store
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or str(settings.vector_store_dir)
        self.embeddings = EmbeddingsFactory.create_embeddings(embeddings_provider)

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"Vector store initialized at {self.persist_directory}")

    def _get_vector_store(self) -> Chroma:
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Append documents to the vector store (prefer replace_documents for full reindex)."""
        if not documents:
            logger.warning("No documents to add to vector store")
            return

        try:
            logger.info(f"Adding {len(documents)} documents to vector store")
            vector_store = self._get_vector_store()
            ids = [_chunk_id(doc, i) for i, doc in enumerate(documents)]
            vector_store.add_documents(documents, ids=ids)
            logger.info(f"Successfully added {len(documents)} documents to vector store")

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def replace_documents(self, documents: List[Document]) -> None:
        """Replace the entire collection with fresh document chunks."""
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
        """
        Get retriever from vector store

        Args:
            search_kwargs: Search parameters (k=number of results, etc)

        Returns:
            Retriever instance
        """
        if search_kwargs is None:
            search_kwargs = {"k": settings.retrieval_k}

        try:
            logger.info(f"Creating retriever with search_kwargs: {search_kwargs}")
            retriever = self._get_vector_store().as_retriever(search_kwargs=search_kwargs)
            logger.info("Retriever created successfully")
            return retriever

        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Direct similarity search against the vector store."""
        k = k or settings.retrieval_k
        return self._get_vector_store().similarity_search(query, k=k)

    def delete_collection(self) -> None:
        """Delete the current collection"""
        try:
            logger.warning(f"Deleting collection: {self.collection_name}")
            self._get_vector_store().delete_collection()
            logger.info("Collection deleted successfully")

        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def get_collection_info(self) -> dict:
        """
        Get information about the collection

        Returns:
            Collection information
        """
        try:
            collection = self._get_vector_store()._collection
            info = {
                "name": self.collection_name,
                "count": collection.count(),
                "persist_directory": self.persist_directory,
            }
            logger.info(f"Collection info: {info}")
            return info

        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
