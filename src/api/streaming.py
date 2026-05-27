"""Utilitários async para streaming RAG via WebSocket / Chainlit."""
import asyncio
from typing import Any, AsyncIterator, Dict

from src.modules.rag_chain import RAGChain


async def stream_rag_events(
    rag_chain: RAGChain,
    question: str,
    conversation_context: str = "",
) -> AsyncIterator[Dict[str, Any]]:
    """Gera eventos de status, meta, tokens e done para clientes streaming."""
    loop = asyncio.get_event_loop()

    from src.modules.llm_factory import LLMFactory

    yield {"type": "status", "message": f"Consultando CDC e {LLMFactory.display_name()}..."}

    ctx = await loop.run_in_executor(
        None,
        rag_chain.prepare_retrieval,
        question,
        conversation_context,
    )

    yield {
        "type": "meta",
        "meta": ctx["meta"],
        "sources": ctx["sources"],
    }

    queue: asyncio.Queue = asyncio.Queue()

    def _run_stream():
        try:
            for token in rag_chain.stream_answer_from_context(ctx):
                loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

    loop.run_in_executor(None, _run_stream)

    full_answer = ""
    while True:
        kind, payload = await queue.get()
        if kind == "token":
            full_answer += payload
            yield {"type": "token", "content": payload}
        elif kind == "done":
            break
        elif kind == "error":
            yield {"type": "error", "message": payload}
            return

    yield {
        "type": "done",
        "answer": full_answer,
        "meta": ctx["meta"],
        "sources": ctx["sources"],
    }
