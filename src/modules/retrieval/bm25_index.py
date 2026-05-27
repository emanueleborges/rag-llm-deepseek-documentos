"""
Índice BM25 persistido em disco (complementa busca vetorial Chroma).
"""
import pickle
import re
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.logger import logger


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def document_key(doc: Document) -> str:
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    preview = doc.page_content[:120]
    return f"{source}|{page}|{preview}"


class BM25Index:
    """Índice BM25 com persistência pickle."""

    def __init__(self, persist_path: Path):
        self.persist_path = Path(persist_path)
        self.documents: List[Document] = []
        self._bm25: Optional[BM25Okapi] = None

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and bool(self.documents)

    def build(self, documents: List[Document]) -> None:
        if not documents:
            self.documents = []
            self._bm25 = None
            self.save()
            return

        self.documents = documents
        corpus = [tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(corpus)
        self.save()
        logger.info("BM25 index built with %d chunks", len(documents))

    def save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [
                {"page_content": d.page_content, "metadata": dict(d.metadata)}
                for d in self.documents
            ],
            "corpus": [tokenize(d.page_content) for d in self.documents],
        }
        with open(self.persist_path, "wb") as handle:
            pickle.dump(payload, handle)

    def load(self) -> bool:
        if not self.persist_path.exists():
            return False
        try:
            with open(self.persist_path, "rb") as handle:
                payload = pickle.load(handle)
            self.documents = [
                Document(page_content=item["page_content"], metadata=item["metadata"])
                for item in payload.get("documents", [])
            ]
            corpus = payload.get("corpus") or [tokenize(d.page_content) for d in self.documents]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            logger.info("BM25 index loaded (%d chunks)", len(self.documents))
            return self.is_ready
        except Exception as exc:
            logger.warning("Failed to load BM25 index: %s", exc)
            self.documents = []
            self._bm25 = None
            return False

    def search(self, query: str, k: int = 20) -> List[Document]:
        if not self.is_ready:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results: List[Document] = []
        for idx, score in ranked[: k * 2]:
            if score <= 0:
                break
            doc = self.documents[idx]
            doc.metadata["bm25_score"] = float(score)
            results.append(doc)
            if len(results) >= k:
                break
        return results

    def clear(self) -> None:
        self.documents = []
        self._bm25 = None
        if self.persist_path.exists():
            self.persist_path.unlink(missing_ok=True)
