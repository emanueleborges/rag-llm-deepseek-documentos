"""Grafo LangGraph — orquestração agentic do RAG."""
from typing import TYPE_CHECKING, Dict, Any

from langgraph.graph import END, StateGraph

from src.agents.state import RAGState
from src.agents.nodes import (
    make_analyze_node,
    make_generate_node,
    make_grade_node,
    make_retrieve_node,
    make_rewrite_node,
    make_web_search_node,
)
from src.config import settings
from src.logger import logger

if TYPE_CHECKING:
    from src.modules.rag_chain import RAGChain


def _route_after_grade(state: RAGState) -> str:
    return state.get("next_step") or "generate"


def build_rag_graph(rag: "RAGChain"):
    workflow = StateGraph(RAGState)

    workflow.add_node("analyze_query", make_analyze_node(rag))
    workflow.add_node("retrieve", make_retrieve_node(rag))
    workflow.add_node("grade_context", make_grade_node(rag))
    workflow.add_node("rewrite_query", make_rewrite_node(rag))
    workflow.add_node("web_search", make_web_search_node(rag))
    workflow.add_node("generate", make_generate_node(rag))

    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", "grade_context")
    workflow.add_conditional_edges(
        "grade_context",
        _route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
            "web_search": "web_search",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_rag_graph(
    rag: "RAGChain",
    question: str,
    conversation_context: str = "",
) -> Dict[str, Any]:
    graph = build_rag_graph(rag)
    initial: RAGState = {
        "question": question,
        "conversation_context": conversation_context,
        "retrieval_attempts": 0,
        "graph_trace": [],
        "documents": [],
    }
    final = graph.invoke(initial)

    docs = final.get("documents") or []
    mode = final.get("mode") or "fallback"
    note = final.get("relevance_note") or ""
    trace = final.get("graph_trace") or []

    meta = rag._build_meta(docs, mode, note)
    meta["langgraph"] = True
    meta["graph_trace"] = trace
    if final.get("web_context"):
        meta["web_enriched"] = True

    logger.info("LangGraph completed | trace=%s", " → ".join(trace))

    return {
        "answer": final.get("answer", ""),
        "sources": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ],
        "meta": meta,
    }
