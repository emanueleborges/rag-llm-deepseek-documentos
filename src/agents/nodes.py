"""
Nós do grafo LangGraph — analisar → buscar → avaliar → (reformular/web) → gerar.
"""
from typing import TYPE_CHECKING, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.agents.state import RAGState
from src.agents.web_search import search_web
from src.config import settings
from src.logger import logger

if TYPE_CHECKING:
    from src.modules.rag_chain import RAGChain

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Reescreva a pergunta do usuário para busca semântica no CDC (Lei 8.078/1990). "
        "Use termos jurídicos, artigos ou conceitos do direito do consumidor. "
        "Retorne APENAS a query reescrita, sem explicação.",
    ),
    ("human", "Pergunta original: {question}\n\nQuery para busca:"),
])


def _trace(state: RAGState, step: str) -> List[str]:
    trace = list(state.get("graph_trace") or [])
    trace.append(step)
    return trace


def make_analyze_node(rag: "RAGChain"):
    def analyze_query(state: RAGState) -> RAGState:
        question = state["question"]
        queries = [question]
        if rag._is_overview_question(question):
            queries.extend([
                "Código de Defesa do Consumidor Lei 8078 objetivo política nacional relações de consumo",
                "definição consumidor fornecedor produto serviço artigo 2 artigo 3",
                "direitos básicos do consumidor artigo 6 CDC",
                "CÓDIGO DE PROTEÇÃO E DEFESA DO CONSUMIDOR",
            ])
        return {
            "queries": queries,
            "retrieval_attempts": state.get("retrieval_attempts", 0),
            "graph_trace": _trace(state, "analyze_query"),
        }

    return analyze_query


def make_retrieve_node(rag: "RAGChain"):
    def retrieve(state: RAGState) -> RAGState:
        queries = state.get("queries") or [state["question"]]
        seen: set = set()
        merged: List[Document] = []
        per_query_k = max(12, settings.hybrid_fetch_k // max(len(queries), 1))
        overview = rag._is_overview_question(state["question"])

        for q in queries:
            for doc in rag.vector_store_manager.advanced_search(
                q,
                k=per_query_k,
                rerank_top_k=settings.rerank_top_k + (2 if overview else 0),
            ):
                key = doc.page_content[:300]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        top_k = settings.rerank_top_k + (4 if overview else 0)
        docs = merged[:top_k]
        logger.info("LangGraph retrieve: %d document(s)", len(docs))
        return {
            "documents": docs,
            "graph_trace": _trace(state, "retrieve"),
        }

    return retrieve


def make_grade_node(rag: "RAGChain"):
    def grade_context(state: RAGState) -> RAGState:
        question = state["question"]
        docs = state.get("documents") or []
        mode, note = rag._assess_retrieval_mode(question, docs)
        attempts = state.get("retrieval_attempts", 0)

        next_step = "generate"
        if not docs and attempts < settings.max_retrieval_retries:
            next_step = "rewrite"
        elif mode == "fallback" and not docs and settings.tavily_api_key.strip():
            next_step = "web_search"
        elif mode == "fallback":
            next_step = "generate"

        return {
            "mode": mode,
            "relevance_note": note,
            "next_step": next_step,
            "graph_trace": _trace(state, f"grade:{mode}->{next_step}"),
        }

    return grade_context


def make_rewrite_node(rag: "RAGChain"):
    def rewrite_query(state: RAGState) -> RAGState:
        question = state["question"]
        messages = REWRITE_PROMPT.format_messages(question=question)
        rewritten = str(rag.llm.invoke(messages)).strip()
        if not rewritten:
            rewritten = question

        attempts = state.get("retrieval_attempts", 0) + 1
        logger.info("LangGraph rewrite (attempt %d): %s", attempts, rewritten[:80])
        return {
            "queries": [rewritten],
            "retrieval_attempts": attempts,
            "graph_trace": _trace(state, "rewrite_query"),
        }

    return rewrite_query


def make_web_search_node(rag: "RAGChain"):
    def web_search_node(state: RAGState) -> RAGState:
        snippet = search_web(state["question"])
        mode = "fallback"
        note = state.get("relevance_note") or ""
        if snippet:
            note = f"{note} Complemento web (Tavily)."
        else:
            note = f"{note} Web search indisponível ou sem resultados."
        return {
            "web_context": snippet,
            "mode": mode,
            "relevance_note": note.strip(),
            "next_step": "generate",
            "graph_trace": _trace(state, "web_search"),
        }

    return web_search_node


def make_generate_node(rag: "RAGChain"):
    def generate(state: RAGState) -> RAGState:
        question = state["question"]
        docs = list(state.get("documents") or [])
        mode = state.get("mode") or "fallback"
        web_context = state.get("web_context") or ""

        context = rag._format_context(docs) if docs else ""
        if web_context:
            context += (
                "\n\n---\n\n[Complemento web — não substitui o CDC indexado]\n"
                + web_context
            )
        if not context.strip():
            context = "(Nenhum trecho relevante recuperado da base indexada.)"

        conv = (state.get("conversation_context") or "").strip()
        if conv:
            question = f"Contexto da conversa:\n{conv}\n\nPergunta atual: {question}"

        template = rag.prompt if mode == "strict" else rag.fallback_prompt
        messages = template.format_messages(context=context, question=question)
        answer = str(rag.llm.invoke(messages))

        return {
            "answer": answer,
            "graph_trace": _trace(state, f"generate:{mode}"),
        }

    return generate
