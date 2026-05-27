"""
Parser PDF com PyMuPDF — layout por página, seções e tipo de conteúdo.
"""
from pathlib import Path
from typing import List

import fitz

from langchain_core.documents import Document

from src.logger import logger
from src.modules.parsers.metadata import (
    base_metadata,
    detect_content_type,
    detect_section,
)


def load_pdf_pymupdf(file_path: Path) -> List[Document]:
    """Extrai texto página a página com metadados enriquecidos."""
    documents: List[Document] = []
    current_section = None

    with fitz.open(file_path) as pdf:
        meta = base_metadata(file_path, parser="pymupdf")
        meta["page_count"] = pdf.page_count

        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            text = page.get_text("text").strip()
            if not text:
                continue

            current_section = detect_section(text, current_section)
            page_meta = {
                **meta,
                "page": page_index,
                "page_label": page_index + 1,
                "section": current_section or "",
                "content_type": detect_content_type(text),
            }
            documents.append(Document(page_content=text, metadata=page_meta))

    logger.info(
        "PyMuPDF: %s → %d página(s) com texto",
        file_path.name,
        len(documents),
    )
    return documents
