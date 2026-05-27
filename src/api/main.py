"""
FastAPI application for RAG Agent
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from src.logger import logger
from src.config import settings
from src.api.streaming import stream_rag_events
from src.modules.feedback_store import feedback_store
from src.modules import (
    DocumentProcessor,
    VectorStoreManager,
    RAGChain,
)


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    conversation_context: str = Field("", description="Short conversation history")
    streaming: bool = Field(False, description="Enable streaming response")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[dict] = Field(default_factory=list, description="Source documents")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Retrieval metadata")


class IngestRequest(BaseModel):
    documents_path: Optional[str] = Field(None, description="Path to documents")


class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=-1, le=1, description="1 = thumbs up, -1 = thumbs down")
    question: str = Field(..., description="User question")
    answer: str = Field(..., description="Assistant answer")
    session_id: str = Field("", description="Client session id")
    message_id: str = Field("", description="Message id")
    meta: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(default="1.0.0", description="API version")


app = FastAPI(
    title="RAG Agent API",
    description="Retrieval-Augmented Generation Agent with Deepseek and LangChain",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain: Optional[RAGChain] = None
vector_store_manager: Optional[VectorStoreManager] = None
document_processor: Optional[DocumentProcessor] = None


def initialize_rag():
    global rag_chain, vector_store_manager, document_processor

    logger.info("Initializing RAG components")
    settings.create_directories()
    vector_store_manager = VectorStoreManager()
    document_processor = DocumentProcessor()
    rag_chain = RAGChain(vector_store_manager)
    logger.info("RAG components initialized successfully")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up RAG Agent API")
    initialize_rag()
    logger.info("RAG Agent API started")


@app.on_event("shutdown")
async def shutdown_event():
    from src.observability.tracer import flush_traces

    flush_traces()
    logger.info("Shutting down RAG Agent API")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    try:
        collection_info = vector_store_manager.get_collection_info()
        status = "healthy" if collection_info else "degraded"
        return HealthResponse(status=status, version="1.1.0")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        if not rag_chain:
            raise HTTPException(status_code=503, detail="RAG system not initialized")

        logger.info(f"Received query: {request.question}")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: rag_chain.query(
                request.question,
                conversation_context=request.conversation_context,
            ),
        )

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            meta=result.get("meta", {}),
        )

    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket para chat streaming.

    Cliente envia:
      {"type": "query", "question": "...", "conversation_context": "..."}
      {"type": "feedback", "rating": 1, "question": "...", "answer": "...", ...}

    Servidor envia:
      {"type": "status"|"meta"|"token"|"done"|"error"|"feedback_saved", ...}
    """
    await websocket.accept()
    if not rag_chain:
        await websocket.send_json({"type": "error", "message": "RAG not initialized"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "query":
                question = (data.get("question") or "").strip()
                if not question:
                    await websocket.send_json(
                        {"type": "error", "message": "question is required"}
                    )
                    continue

                conv_ctx = data.get("conversation_context") or ""
                async for event in stream_rag_events(rag_chain, question, conv_ctx):
                    await websocket.send_json(event)

            elif msg_type == "feedback":
                rating = int(data.get("rating", 0))
                if rating not in (1, -1):
                    await websocket.send_json(
                        {"type": "error", "message": "rating must be 1 or -1"}
                    )
                    continue
                entry_id = feedback_store.save(
                    rating=rating,
                    question=data.get("question", ""),
                    answer=data.get("answer", ""),
                    session_id=data.get("session_id", ""),
                    message_id=data.get("message_id", ""),
                    meta=data.get("meta") or {},
                    source="websocket",
                )
                await websocket.send_json(
                    {"type": "feedback_saved", "id": entry_id}
                )

            else:
                await websocket.send_json(
                    {"type": "error", "message": f"unknown type: {msg_type}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest) -> dict:
    if request.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating must be 1 or -1")

    entry_id = feedback_store.save(
        rating=request.rating,
        question=request.question,
        answer=request.answer,
        session_id=request.session_id,
        message_id=request.message_id,
        meta=request.meta,
        source="api",
    )
    return {"status": "ok", "id": entry_id}


@app.get("/feedback")
async def list_feedback(limit: int = 50) -> dict:
    return {"items": feedback_store.list_recent(limit=limit)}


@app.post("/ingest")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> dict:
    try:
        if not document_processor or not vector_store_manager:
            raise HTTPException(status_code=503, detail="System not initialized")

        documents_path = request.documents_path or str(settings.documents_dir)
        logger.info(f"Starting document ingestion from {documents_path}")
        background_tasks.add_task(_ingest_documents, documents_path)
        return {"status": "ingestion_started", "path": documents_path}

    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingest error: {str(e)}")


async def _ingest_documents(documents_path: str):
    try:
        logger.info(f"Ingesting documents from {documents_path}")
        documents = document_processor.process_documents(documents_path)

        if not documents:
            logger.warning("No documents found to ingest")
            return

        vector_store_manager.replace_documents(documents)
        global rag_chain
        rag_chain = RAGChain(vector_store_manager)
        logger.info(f"Successfully ingested {len(documents)} document chunks")

    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")


@app.get("/collection-info")
async def get_collection_info() -> dict:
    try:
        if not vector_store_manager:
            raise HTTPException(status_code=503, detail="System not initialized")
        return vector_store_manager.get_collection_info()
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail="Error getting collection info")


@app.delete("/collection")
async def delete_collection() -> dict:
    try:
        if not vector_store_manager:
            raise HTTPException(status_code=503, detail="System not initialized")

        vector_store_manager.delete_collection()
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
