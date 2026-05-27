"""
Factory de parsers PDF — LlamaParse (se configurado) ou PyMuPDF/pypdf.
"""
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader

from src.config import settings
from src.logger import logger
from src.modules.parsers.llama_parser import load_pdf_llamaparse
from src.modules.parsers.metadata import enrich_document


def _load_pdf_pypdf(file_path: Path) -> List[Document]:
    reader = PdfReader(str(file_path))
    docs: List[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        docolumn.append(
            enrich_document(
                Document(
                    page_content=text,
                    metadata={"source": str(file_path), "page": page_num},
                ),
                parser="pypdf",
            )
        )
    return docs


def load_pdf(file_path: Path) -> List[Document]:
    """Carrega um PDF com o melhor parser disponível."""
    backend = (settings.pdf_parser or "auto").lower()
    api_key = settings.llama_cloud_api_key.strip()

    if backend in ("llamaparse", "llama_parse", "auto") and api_key:
        try:
            return load_pdf_llamaparse(file_path, api_key)
        except Exception as exc:
            logger.warning(
                "LlamaParse falhou para %s (%s); usando fallback local.",
                file_path.name,
                exc,
            )

    if backend in ("pymupdf", "auto"):
        try:
            from src.modules.parsers.pymupdf_parser import load_pdf_pymupdf

            return load_pdf_pymupdf(file_path)
        except ImportError as exc:
            logger.warning(
                "PyMuPDF não instalado (%s); usando pypdf. "
                "Instale com: pip install pymupdf",
                exc,
            )
        except Exception as exc:
            logger.warning(
                "PyMuPDF falhou para %s (%s); usando pypdf.",
                file_path.name,
                exc,
            )

    return _load_pdf_pypdf(file_path)
