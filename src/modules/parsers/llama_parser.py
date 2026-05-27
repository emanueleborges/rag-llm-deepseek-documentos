"""
Parser PDF via LlamaParse (opcional) — tabelas e layout complexo.
Requer LLAMA_CLOUD_API_KEY no .env.
"""
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.logger import logger
from src.modules.parsers.metadata import base_metadata, detect_content_type, detect_section


def load_pdf_llamaparse(file_path: Path, api_key: str) -> List[Document]:
    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        language="pt",
        verbose=False,
    )
    parsed = parser.load_data(str(file_path))
    if not parsed:
        return []

    base = base_metadata(file_path, parser="llamaparse")
    documents: List[Document] = []
    current_section = None

    for i, item in enumerate(parsed):
        text = getattr(item, "text", None) or str(item)
        text = text.strip()
        if not text:
            continue

        item_meta = getattr(item, "metadata", None) or {}
        page = item_meta.get("page", item_meta.get("page_number", i))
        current_section = detect_section(text, current_section)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    **base,
                    "page": page if isinstance(page, int) else i,
                    "page_label": (page + 1) if isinstance(page, int) else i + 1,
                    "section": current_section or "",
                    "content_type": detect_content_type(text),
                },
            )
        )

    logger.info(
        "LlamaParse: %s → %d bloco(s)",
        file_path.name,
        len(documents),
    )
    return documents
