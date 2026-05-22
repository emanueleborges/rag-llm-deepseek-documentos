"""
Example script demonstrating RAG Agent usage
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.logger import logger
from src.config import settings
from src.modules import (
    DocumentProcessor,
    VectorStoreManager,
    RAGChain,
)


def example_basic_workflow():
    """Example: Basic RAG workflow"""
    logger.info("=" * 60)
    logger.info("Example: Basic RAG Workflow")
    logger.info("=" * 60)

    # Initialize components
    logger.info("\n1. Initializing components...")
    vector_store_manager = VectorStoreManager()
    document_processor = DocumentProcessor()
    rag_chain = RAGChain(vector_store_manager)

    logger.info("✓ Components initialized")

    # Process documents
    logger.info("\n2. Processing documents...")
    documents = document_processor.process_documents()

    if documents:
        logger.info(f"✓ Loaded {len(documents)} document chunks")

        # Add to vector store
        logger.info("\n3. Adding documents to vector store...")
        vector_store_manager.replace_documents(documents)
        logger.info("✓ Documents added to vector store")

        # Reinitialize RAG chain
        logger.info("\n4. Reinitializing RAG chain...")
        rag_chain = RAGChain(vector_store_manager)
        logger.info("✓ RAG chain reinitialized")

        # Query examples
        logger.info("\n5. Running queries...")
        questions = [
            "What is RAG?",
            "How does LangChain work?",
            "What are embeddings?",
        ]

        for question in questions:
            logger.info(f"\nQuestion: {question}")
            try:
                result = rag_chain.query(question)
                logger.info(f"Answer: {result['answer'][:200]}...")
                logger.info(f"Sources found: {len(result['sources'])}")
            except Exception as e:
                logger.error(f"Error: {e}")

    else:
        logger.warning("No documents found. Please add documents to ./data/documents/")


def example_batch_processing():
    """Example: Batch query processing"""
    logger.info("=" * 60)
    logger.info("Example: Batch Query Processing")
    logger.info("=" * 60)

    # Initialize
    vector_store_manager = VectorStoreManager()
    rag_chain = RAGChain(vector_store_manager)

    # Batch queries
    questions = [
        "What is the main topic?",
        "Who are the key people?",
        "What are the main challenges?",
    ]

    logger.info(f"\nProcessing {len(questions)} queries...")

    results = rag_chain.batch_query(questions)

    for i, (question, result) in enumerate(zip(questions, results), 1):
        logger.info(f"\nQuery {i}: {question}")
        logger.info(f"Answer: {result['answer'][:100]}...")


def example_collection_management():
    """Example: Collection management"""
    logger.info("=" * 60)
    logger.info("Example: Collection Management")
    logger.info("=" * 60)

    vector_store_manager = VectorStoreManager()

    # Get collection info
    logger.info("\nGetting collection information...")
    info = vector_store_manager.get_collection_info()
    logger.info(f"Collection info: {info}")

    # Note: Uncomment to delete collection
    # logger.info("\nDeleting collection...")
    # vector_store_manager.delete_collection()
    # logger.info("✓ Collection deleted")


if __name__ == "__main__":
    # Create example directory structure
    settings.create_directories()

    # Run examples
    try:
        example_basic_workflow()
        # example_batch_processing()
        # example_collection_management()

    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Execution error: {e}")
        raise
