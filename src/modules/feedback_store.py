"""Persistência de feedback 👍/👎 das respostas."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import settings
from src.logger import logger


class FeedbackStore:
    """Armazena feedback em JSONL (append-only)."""

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = Path(store_dir or settings.feedback_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.store_dir / "feedback.jsonl"

    def save(
        self,
        rating: int,
        question: str,
        answer: str,
        *,
        session_id: str = "",
        message_id: str = "",
        meta: Optional[Dict[str, Any]] = None,
        source: str = "api",
    ) -> str:
        entry_id = message_id or str(uuid.uuid4())
        record = {
            "id": entry_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rating": rating,
            "question": question,
            "answer": answer[:4000],
            "session_id": session_id,
            "source": source,
            "meta": meta or {},
        }
        with self.file_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Feedback saved | rating=%s | id=%s | source=%s", rating, entry_id, source)
        return entry_id

    def list_recent(self, limit: int = 50) -> list:
        if not self.file_path.exists():
            return []
        lines = self.file_path.read_text(encoding="utf-8").strip().splitlines()
        records = []
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


feedback_store = FeedbackStore()
