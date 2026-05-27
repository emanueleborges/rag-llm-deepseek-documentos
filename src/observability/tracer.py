"""Observabilidade — traces Langfuse (opcional)."""
from typing import Any, Dict, Optional

from src.config import settings
from src.logger import logger

_langfuse_client = None


def _is_configured() -> bool:
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key.strip()
        and settings.langfuse_secret_key.strip()
    )


def get_langfuse():
    """Retorna cliente Langfuse singleton ou None."""
    global _langfuse_client
    if not _is_configured():
        return None
    if _langfuse_client is not None:
        return _langfuse_client
    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse tracing enabled (%s)", settings.langfuse_host)
        return _langfuse_client
    except ImportError:
        logger.warning("Langfuse SDK not installed — tracing disabled")
        return None
    except Exception as exc:
        logger.warning("Langfuse init failed: %s", exc)
        return None


def trace_rag_query(
    question: str,
    result: Dict[str, Any],
    duration_ms: float,
    conversation_context: str = "",
) -> None:
    """Envia trace de uma consulta RAG ao Langfuse (no-op se desabilitado)."""
    client = get_langfuse()
    if client is None:
        return

    meta = result.get("meta") or {}
    sources = result.get("sources") or []
    contexts = [s.get("content", "")[:500] for s in sources[:5]]

    try:
        trace = client.trace(
            name="rag-query",
            input={
                "question": question,
                "conversation_context": conversation_context[:500],
            },
            metadata={
                "environment": settings.environment,
                "mode": meta.get("mode"),
                "sources_count": meta.get("sources_count"),
                "grounded": meta.get("grounded"),
                "fallback": meta.get("fallback"),
                "graph_trace": meta.get("graph_trace"),
                "duration_ms": round(duration_ms, 2),
            },
            tags=[settings.environment, meta.get("mode", "unknown")],
        )

        trace.span(
            name="retrieval",
            input={"question": question},
            output={"contexts_preview": contexts, "meta": meta},
        )

        from src.modules.llm_factory import LLMFactory

        trace.generation(
            name="llm-answer",
            model=LLMFactory.active_model_name(),
            input=question,
            output=result.get("answer", ""),
            metadata={
                "retrieval_note": meta.get("retrieval_note"),
                "source_files": meta.get("source_files"),
            },
        )

        client.flush()
    except Exception as exc:
        logger.warning("Langfuse trace failed: %s", exc)


def flush_traces() -> None:
    client = get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)
