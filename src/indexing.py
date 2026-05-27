"""Vector index versioning and reindex helpers."""
from pathlib import Path

from src.config import DEFAULT_INDEX_VERSION, settings, current_index_version
from src.logger import logger
from src.modules.document_processor import DocumentProcessor
from src.modules.vector_store import VectorStoreManager


def _marker_path() -> Path:
    return settings.vector_store_dir / ".embedding_version"


def needs_reindex() -> bool:
    """True when embeddings model changed or index never built."""
    marker = _marker_path()
    if not marker.exists():
        return True
    stored = marker.read_text(encoding="utf-8").strip()
    return stored != current_index_version()


def mark_indexed() -> None:
    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    _marker_path().write_text(current_index_version(), encoding="utf-8")


def reindex_documents(
    vector_store_manager: VectorStoreManager,
    document_processor: DocumentProcessor = None,
) -> int:
    """Rebuild the vector collection from files in data/documents."""
    processor = document_processor or DocumentProcessor()
    documents = processor.process_documents()

    if not documents:
        logger.warning("No documents to index")
        return 0

    if hasattr(vector_store_manager, "replace_documents"):
        vector_store_manager.replace_documents(documents)
    else:
        vector_store_manager.delete_collection()
        vector_store_manager.add_documents(documents)

    mark_indexed()
    logger.info(f"Indexed {len(documents)} chunks")
    return len(documents)
