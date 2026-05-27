"""
Reranker semântico BGE (Cross-Encoder) — Fase 2.
"""
from typing import List, Optional

from langchain_core.documents import Document

from src.config import settings
from src.logger import logger


class BGEReranker:
    """Reordena candidatos com cross-encoder multilíngue."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers não instalado — defina RERANKER_ENABLED=false "
                    "ou instale: pip install -r requirements.txt"
                ) from exc

            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None,
        min_score: float = None,
    ) -> List[Document]:
        if not documents:
            return []

        top_k = top_k or settings.rerank_top_k
        min_score = settings.reranker_min_score if min_score is None else min_score

        if len(documents) <= top_k:
            return documents

        model = self._get_model()
        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(scores, documents),
            key=lambda item: float(item[0]),
            reverse=True,
        )

        results: List[Document] = []
        for score, doc in ranked:
            score_f = float(score)
            if min_score is not None and score_f < min_score:
                continue
            doc.metadata["rerank_score"] = round(score_f, 4)
            results.append(doc)
            if len(results) >= top_k:
                break

        return results or [doc for _, doc in ranked[:top_k]]
