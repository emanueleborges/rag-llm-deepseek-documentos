"""
Utilitários de metadados e detecção de estrutura em documentos jurídicos.
"""
import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

SECTION_PATTERNS = (
    re.compile(r"^(TÍTULO|CAPÍTULO|SEÇÃO|Seção)\s", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Art\.\s*\d+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^ANEXO\s", re.IGNORECASE | re.MULTILINE),
)


def detect_section(text: str, current: Optional[str] = None) -> Optional[str]:
    """Atualiza a seção corrente a partir do texto da página/chunk."""
    section = current
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in SECTION_PATTERNS:
            if pattern.match(stripped):
                section = stripped[:160]
                break
    return section


def detect_content_type(text: str) -> str:
    """Heurística simples: texto corrido vs. trecho tabular."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return "text"

    table_like = 0
    for line in lines:
        if line.count("|") >= 2 or line.count("\t") >= 2:
            table_like += 1
        elif re.search(r"\s{3,}\S+\s{3,}\S+", line):
            table_like += 1

    if table_like >= 2 or (table_like / len(lines)) > 0.35:
        return "table"
    return "text"


def base_metadata(path: Path, parser: str) -> dict:
    return {
        "source": str(path.resolve()),
        "filename": path.name,
        "file_type": path.suffix.lstrip(".").lower() or "unknown",
        "parser": parser,
    }


def enrich_document(doc: Document, parser: str) -> Document:
    """Garante metadados mínimos em qualquer Document."""
    source = doc.metadata.get("source", "")
    path = Path(source) if source else None
    if path and path.exists():
        doc.metadata.setdefault("filename", path.name)
        doc.metadata.setdefault("file_type", path.suffix.lstrip(".").lower())
    doc.metadata.setdefault("parser", parser)
    doc.metadata.setdefault("content_type", detect_content_type(doc.page_content))
    if "section" not in doc.metadata:
        doc.metadata["section"] = detect_section(doc.page_content) or ""
    return doc
