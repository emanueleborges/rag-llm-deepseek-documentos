"""
Document loader and processor for RAG Agent
Fase 1: parsing avançado + metadados enriquecidos (página, seção, tipo).
"""
from collections import defaultdict
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.logger import logger
from src.config import settings
from src.modules.parsers.metadata import enrich_document
from src.modules.parsers.pdf_loader import load_pdf


class DocumentProcessor:
    """Process and split documents for RAG"""

    def __init__(
        self,
        chunk_size: int = settings.max_chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def _load_pdf_files(self, docs_path: Path) -> List[Document]:
        documents: List[Document] = []
        for pdf_path in sorted(docs_path.rglob("*.pdf")):
            try:
                documents.extend(load_pdf(pdf_path))
            except Exception as exc:
                logger.warning("Erro ao carregar PDF %s: %s", pdf_path.name, exc)
        logger.info("Carregados %d bloco(s) de PDF", len(documents))
        return documents

    def load_documents(self, documents_path: str = None) -> List[Document]:
        if documents_path is None:
            documents_path = str(settings.documents_dir)

        docs_path = Path(documents_path)
        if not docs_path.exists():
            logger.warning(f"Documents directory not found: {documents_path}")
            return []

        documents = self._load_pdf_files(docs_path)

        for txt_path in sorted(docs_path.rglob("*.txt")):
            try:
                content = txt_path.read_text(encoding="utf-8")
                doc = enrich_document(
                    Document(
                        page_content=content,
                        metadata={"source": str(txt_path)},
                    ),
                    parser="text",
                )
                documents.append(doc)
            except Exception as exc:
                logger.warning("Erro ao carregar TXT %s: %s", txt_path.name, exc)

        if documents:
            logger.info("Loaded %d document block(s) including TXT", len(documents))

        if not documents:
            logger.warning("No documents found to load")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            logger.warning("No documents to split")
            return []

        split_docs = self.text_splitter.split_documents(documents)
        chunk_counters: defaultdict[str, int] = defaultdict(int)
        source_totals: defaultdict[str, int] = defaultdict(int)

        for chunk in split_docs:
            source = chunk.metadata.get("source", "unknown")
            source_totals[source] += 1

        for chunk in split_docs:
            source = chunk.metadata.get("source", "unknown")
            if chunk.metadata.get("source"):
                chunk.metadata["filename"] = Path(chunk.metadata["source"]).name

            idx = chunk_counters[source]
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_id"] = (
                f"{Path(source).stem if source != 'unknown' else 'doc'}_{idx}"
            )
            chunk.metadata["chunks_in_source"] = source_totals[source]
            chunk_counters[source] += 1

            chunk.metadata.setdefault("content_type", "text")
            chunk.metadata.setdefault("section", "")

        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs

    def process_documents(self, documents_path: str = None) -> List[Document]:
        logger.info(
            "Processing documents from %s (parser=%s)",
            documents_path or settings.documents_dir,
            settings.pdf_parser,
        )
        documents = self.load_documents(documents_path)
        return self.split_documents(documents)
