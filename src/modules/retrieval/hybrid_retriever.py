"""
Fusão BM25 + busca vetorial (Reciprocal Rank Fusion).
"""
from typing import Callable, List, Optional

from langchain_core.documents import Document

from src.config import settings
from src.logger import logger
from src.modules.retrieval.bm25_index import BM25Index, document_key


class HybridRetriever:
    """Combina resultados vetoriais e BM25 via RRF."""

    def __init__(
        self,
        vector_search: Callable[..., List[Document]],
        bm25_index: BM25Index,
    ):
        self.vector_search = vector_search
        self.bm25_index = bm25_index

    def search(
        self,
        query: str,
        k: int = None,
        metadata_filter: dict = None,
    ) -> List[Document]:
        k = k or settings.hybrid_fetch_k
        vector_k = max(k, settings.retrieval_k)
        bm25_k = max(k, settings.retrieval_k)

        vector_docs = self.vector_search(query, k=vector_k, filter=metadata_filter)
        bm25_docs = (
            self.bm25_index.search(query, k=bm25_k)
            if self.bm25_index.is_ready
            else []
        )

        if not bm25_docs:
            logger.debug("Hybrid search: BM25 unavailable, using vector only")
            return vector_docs[:k]

        fused = self._reciprocal_rank_fusion([vector_docs, bm25_docs], limit=k)
        logger.debug(
            "Hybrid RRF: vector=%d bm25=%d fused=%d",
            len(vector_docs),
            len(bm25_docs),
            len(fused),
        )
        return fused

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[Document]],
        limit: int,
        rrf_k: int = None,
    ) -> List[Document]:
        rrf_k = rrf_k or settings.hybrid_rrf_k
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = document_key(doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
                if key not in doc_map:
                    doc_map[key] = doc

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        results: List[Document] = []
        for key, score in ordered[:limit]:
            doc = doc_map[key]
            doc.metadata["rrf_score"] = round(score, 6)
            results.append(doc)
        return results
