"""
Chainlit UI — streaming, citações clicáveis e feedback (Fase 4).
Execute: chainlit run src/ui/chainlit_app.py --host 0.0.0.0 --port 8502
"""
import uuid
from pathlib import Path
from typing import Any, Dict, List

import chainlit as cl

from src.api.streaming import stream_rag_events
from src.config import settings
from src.indexing import needs_reindex, reindex_documents
from src.logger import logger
from src.modules.document_processor import DocumentProcessor
from src.modules.feedback_store import feedback_store
from src.modules.rag_chain import RAGChain
from src.modules.vector_store import VectorStoreManager

_rag_chain: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        settings.create_directories()
        vsm = VectorStoreManager()
        if needs_reindex():
            logger.info("Chainlit: reindexing outdated vector store")
            reindex_documents(vsm, DocumentProcessor())
        _rag_chain = RAGChain(vsm)
    return _rag_chain


def _conversation_context(history: List[Dict[str, str]]) -> str:
    lines = []
    for turn in history[-4:]:
        lines.append(f"Usuário: {turn['question'][:300]}")
        lines.append(f"Assistente: {turn['answer'][:300]}")
    return "\n".join(lines)


def _source_elements(sources: List[dict]) -> List[cl.Text]:
    elements = []
    for i, src in enumerate(sources, 1):
        meta = src.get("metadata") or {}
        fname = meta.get("filename") or Path(str(meta.get("source", ""))).name or "documento"
        page = meta.get("page_label") or meta.get("page")
        section = meta.get("section") or ""
        title = f"📄 Fonte {i}: {fname}"
        if page is not None:
            title += f" · pág. {page}"
        if section:
            title += f" · {section[:50]}"
        preview = (src.get("content") or "")[:3000]
        elements.append(cl.Text(name=title, content=preview, display="side"))
    return elements


def _save_feedback(action: cl.Action, rating: int) -> None:
    payload = action.payload or {}
    feedback_store.save(
        rating=rating,
        question=payload.get("question", ""),
        answer=payload.get("answer", ""),
        session_id=cl.user_session.get("session_id", ""),
        meta=payload.get("meta") or {},
        source="chainlit",
    )


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    cl.user_session.set("session_id", str(uuid.uuid4()))
    try:
        get_rag_chain()
        welcome = (
            "Olá! Sou o **Assistente CDC** — direito do consumidor com RAG híbrido, "
            "reranker BGE e agente LangGraph.\n\n"
            "Pergunte sobre o Código de Defesa do Consumidor ou procedimentos do Procon."
        )
        if needs_reindex():
            welcome += (
                "\n\n⚠️ O índice vetorial está desatualizado — "
                "reindexe os documentos em `data/documents/`."
            )
        await cl.Message(content=welcome).send()
    except Exception as exc:
        logger.error("Chainlit init error: %s", exc)
        await cl.Message(content=f"Erro ao inicializar RAG: {exc}").send()


@cl.on_message
async def on_message(message: cl.Message):
    rag = get_rag_chain()
    history: List[Dict[str, str]] = cl.user_session.get("history") or []
    conv_ctx = _conversation_context(history)

    msg = cl.Message(content="", author="Assistente CDC")
    await msg.send()

    meta: Dict[str, Any] = {}
    sources: List[dict] = []
    answer = ""

    async with cl.Step(name="Recuperação RAG", type="tool") as step:
        async for event in stream_rag_events(rag, message.content, conv_ctx):
            etype = event.get("type")
            if etype == "status":
                step.output = event.get("message", "")
            elif etype == "meta":
                meta = event.get("meta") or {}
                sources = event.get("sources") or []
                mode = meta.get("mode", "")
                note = meta.get("retrieval_note", "")
                trace = meta.get("graph_trace")
                parts = [f"Modo: **{mode}**"]
                if note:
                    parts.append(note)
                if trace:
                    parts.append(f"Agente: {' → '.join(trace)}")
                step.output = " · ".join(parts)
            elif etype == "token":
                answer += event["content"]
                await msg.stream_token(event["content"])
            elif etype == "done":
                answer = event.get("answer") or answer
                meta = event.get("meta") or meta
                sources = event.get("sources") or sources
            elif etype == "error":
                await cl.Message(content=f"Erro: {event.get('message')}").send()
                return

    msg.elements = _source_elements(sources)
    msg.actions = [
        cl.Action(
            name="feedback_up",
            payload={
                "question": message.content,
                "answer": answer[:4000],
                "meta": meta,
            },
            label="👍",
            description="Resposta útil",
        ),
        cl.Action(
            name="feedback_down",
            payload={
                "question": message.content,
                "answer": answer[:4000],
                "meta": meta,
            },
            label="👎",
            description="Resposta não útil",
        ),
    ]
    await msg.update()

    history.append({"question": message.content, "answer": answer})
    cl.user_session.set("history", history)


@cl.action_callback("feedback_up")
async def on_feedback_up(action: cl.Action):
    _save_feedback(action, 1)
    await cl.Message(content="Obrigado pelo feedback! 👍").send()
    await action.remove()


@cl.action_callback("feedback_down")
async def on_feedback_down(action: cl.Action):
    _save_feedback(action, -1)
    await cl.Message(content="Obrigado pelo feedback — vamos melhorar. 👎").send()
    await action.remove()
