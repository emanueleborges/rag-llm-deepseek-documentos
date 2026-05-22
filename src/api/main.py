"""
FastAPI application for RAG Agent
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio

from src.logger import logger
from src.config import settings
from src.modules import (
    DocumentProcessor,
    VectorStoreManager,
    RAGChain,
)


# Pydantic models for API
class QueryRequest(BaseModel):
    """Query request model"""

    question: str = Field(..., description="Question to answer")
    streaming: bool = Field(False, description="Enable streaming response")


class QueryResponse(BaseModel):
    """Query response model"""

    answer: str = Field(..., description="Generated answer")
    sources: List[dict] = Field(default_factory=list, description="Source documents")


class DocumentUploadRequest(BaseModel):
    """Document upload request"""

    file_path: str = Field(..., description="Path to document file")


class IngestRequest(BaseModel):
    """Ingest request"""

    documents_path: Optional[str] = Field(None, description="Path to documents")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    version: str = Field(default="1.0.0", description="API version")


# Initialize FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="Retrieval-Augmented Generation Agent with Deepseek and LangChain",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG components
rag_chain: Optional[RAGChain] = None
vector_store_manager: Optional[VectorStoreManager] = None
document_processor: Optional[DocumentProcessor] = None


def initialize_rag():
    """Initialize RAG components"""
    global rag_chain, vector_store_manager, document_processor

    logger.info("Initializing RAG components")

    # Initialize vector store manager
    vector_store_manager = VectorStoreManager()

    # Initialize document processor
    document_processor = DocumentProcessor()

    # Initialize RAG chain
    rag_chain = RAGChain(vector_store_manager)

    logger.info("RAG components initialized successfully")


@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("Starting up RAG Agent API")
    initialize_rag()
    logger.info("RAG Agent API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("Shutting down RAG Agent API")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint
    
    Returns:
        Health status
    """
    try:
        collection_info = vector_store_manager.get_collection_info()
        status = "healthy" if collection_info else "degraded"

        return HealthResponse(
            status=status,
            version="1.0.0",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system
    
    Args:
        request: Query request
        
    Returns:
        Query response with answer and sources
    """
    try:
        if not rag_chain:
            raise HTTPException(status_code=503, detail="RAG system not initialized")

        logger.info(f"Received query: {request.question}")

        # Run query in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, rag_chain.query, request.question)

        return QueryResponse(answer=result["answer"], sources=result["sources"])

    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


@app.post("/ingest")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Ingest documents into the vector store
    
    Args:
        request: Ingest request
        background_tasks: Background tasks
        
    Returns:
        Ingest status
    """
    try:
        if not document_processor or not vector_store_manager:
            raise HTTPException(status_code=503, detail="System not initialized")

        documents_path = request.documents_path or str(settings.documents_dir)
        logger.info(f"Starting document ingestion from {documents_path}")

        # Run ingestion in background
        background_tasks.add_task(
            _ingest_documents,
            documents_path,
        )

        return {"status": "ingestion_started", "path": documents_path}

    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingest error: {str(e)}")


async def _ingest_documents(documents_path: str):
    """
    Background task for document ingestion
    
    Args:
        documents_path: Path to documents
    """
    try:
        logger.info(f"Ingesting documents from {documents_path}")

        # Process documents
        documents = document_processor.process_documents(documents_path)

        if not documents:
            logger.warning("No documents found to ingest")
            return

        # Add to vector store
        vector_store_manager.replace_documents(documents)

        # Reinitialize RAG chain with new documents
        global rag_chain
        rag_chain = RAGChain(vector_store_manager)

        logger.info(f"Successfully ingested {len(documents)} document chunks")

    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")


@app.get("/collection-info")
async def get_collection_info() -> dict:
    """
    Get vector store collection information
    
    Returns:
        Collection information
    """
    try:
        if not vector_store_manager:
            raise HTTPException(status_code=503, detail="System not initialized")

        info = vector_store_manager.get_collection_info()
        return info

    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail="Error getting collection info")


@app.delete("/collection")
async def delete_collection() -> dict:
    """
    Delete the vector store collection
    
    Returns:
        Status message
    """
    try:
        if not vector_store_manager:
            raise HTTPException(status_code=503, detail="System not initialized")

        vector_store_manager.delete_collection()

        # Reinitialize RAG chain
        global rag_chain
        rag_chain = RAGChain(vector_store_manager)

        return {"status": "collection_deleted"}

    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail="Error deleting collection")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
    )
