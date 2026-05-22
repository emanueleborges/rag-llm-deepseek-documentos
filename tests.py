"""
Test suite for RAG Agent
"""
import pytest
from pathlib import Path
from src.config import settings
from src.modules import (
    DocumentProcessor,
    VectorStoreManager,
    RAGChain,
    LocalEmbeddings,
)


class TestConfig:
    """Test configuration"""

    def test_settings_loaded(self):
        """Test if settings are loaded correctly"""
        assert settings.deepseek_model == "deepseek-chat"
        assert settings.max_chunk_size == 1000
        assert settings.chunk_overlap == 100

    def test_directories_created(self):
        """Test if directories are created"""
        settings.create_directories()
        assert settings.vector_store_dir.exists()
        assert settings.documents_dir.exists()
        assert settings.logs_dir.exists()


class TestDocumentProcessor:
    """Test document processor"""

    def test_processor_initialization(self):
        """Test processor initialization"""
        processor = DocumentProcessor()
        assert processor.chunk_size == settings.max_chunk_size
        assert processor.chunk_overlap == settings.chunk_overlap

    def test_load_empty_documents(self):
        """Test loading from empty directory"""
        processor = DocumentProcessor()
        # Create a temporary empty directory
        temp_dir = Path("./temp_empty")
        temp_dir.mkdir(exist_ok=True)

        documents = processor.load_documents(str(temp_dir))
        assert isinstance(documents, list)

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestEmbeddings:
    """Test embeddings"""

    def test_embeddings_initialization(self):
        """Test embeddings initialization"""
        embeddings = LocalEmbeddings()
        assert embeddings.model == "deepseek-chat"

    def test_hash_to_embedding(self):
        """Test hash to embedding conversion"""
        embeddings = LocalEmbeddings()
        embedding = embeddings._hash_to_embedding(12345, dim=10)
        assert isinstance(embedding, list)
        assert len(embedding) == 10
        # Check normalization
        norm = sum(x**2 for x in embedding) ** 0.5
        assert abs(norm - 1.0) < 0.01


class TestVectorStoreManager:
    """Test vector store manager"""

    def test_manager_initialization(self):
        """Test manager initialization"""
        manager = VectorStoreManager(collection_name="test_collection")
        assert manager.collection_name == "test_collection"
        assert manager.persist_directory

    def test_get_collection_info(self):
        """Test getting collection info"""
        manager = VectorStoreManager(collection_name="test_collection")
        info = manager.get_collection_info()
        assert isinstance(info, dict)
        assert "name" in info or len(info) == 0  # May be empty on first run


class TestRAGChain:
    """Test RAG chain"""

    def test_rag_initialization(self):
        """Test RAG chain initialization"""
        manager = VectorStoreManager(collection_name="test_collection")
        rag = RAGChain(manager)
        assert rag.llm is not None
        assert rag.retriever is not None
        assert rag.chain is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
